import itertools
from math import exp, log, sqrt, floor
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.decomposition import PCA

from matplotlib import pyplot as plt
from matplotlib import colors as colors
from matplotlib import cm as cmx
import seaborn as sns

from raschpy.base import Rasch


class PCM(Rasch):

    def __init__(self,
                 dataframe,
                 max_score_vector=None,
                 extreme_persons=True,
                 no_of_classes=5):

        if max_score_vector is None:
            self.max_score_vector = pd.Series(
                {item: int(max_score)
                 for item, max_score in zip(dataframe.columns,
                                            dataframe.max().to_numpy())}
            )
        else:
            self.max_score_vector = pd.Series(
                {item: int(max_score)
                 for item, max_score in zip(dataframe.columns, max_score_vector)}
            )

        if extreme_persons:
            self.invalid_responses = dataframe[dataframe.isna().all(axis=1)]
            self.dataframe = dataframe[~dataframe.isna().all(axis=1)]
        else:
            scores = dataframe.sum(axis=1)
            max_scores = dataframe.notna().mul(self.max_score_vector, axis=1).sum(axis=1)
            self.invalid_responses = dataframe[(scores == 0) | (scores == max_scores)]
            self.dataframe = dataframe[(scores > 0) & (scores < max_scores)]

        self.no_of_items = self.dataframe.shape[1]
        self.items = self.dataframe.columns
        self.no_of_persons = self.dataframe.shape[0]
        self.persons = self.dataframe.index
        self.no_of_classes = no_of_classes

    '''
    Partial Credit Model (Masters 1982) formulation of the polytomous Rasch model,
    with associated methods.
    '''

    # ------------------------------------------------------------------
    # Core probability / expected-score functions (scalar, used in plots)
    # ------------------------------------------------------------------

    def cat_prob_centred(self, ability, difficulty, category, thresholds):
        """
        Compute the probability of a response category using centred parameterisation.

        Uses the PCM formulation with a central item difficulty and
        Rasch-Andrich threshold offsets. Vectorised using numpy cumsum for
        performance. P(X=k) = exp(k*(b-d) - cumsum(tau)_k) / sum over all categories,
        where b is ability, d is difficulty, tau are thresholds (tau[0]=0 by convention).

        Parameters
        ----------
        ability : float
            Person ability estimate on the logit scale.
        difficulty : float
            Central item difficulty on the logit scale.
        category : int
            Response category (0 to max_score).
        thresholds : array-like
            Rasch-Andrich threshold offsets, length max_score + 1,
            with thresholds[0] = 0 by convention.

        Returns
        -------
        float
            Probability of the specified category, in [0, 1].
        """
        max_score = len(thresholds) - 1
        cats = np.arange(max_score + 1)
        cumsum = np.concatenate(([0.0], np.cumsum(thresholds[1:])))
        log_nums = cats * (ability - difficulty) - cumsum
        log_nums -= log_nums.max()          # numerical stability
        nums = np.exp(log_nums)
        return nums[category] / nums.sum()

    def cat_prob_uncentred(self, ability, category, thresholds):
        """
        Compute the probability of a response category using uncentred parameterisation.

        Uses the PCM formulation with uncentred (absolute) item-category thresholds.
        Numerically stabilised via log-sum-exp. P(X=k) = exp(k*b - cumsum(tau)_k) /
        sum over all categories, where b is ability and tau are uncentred thresholds.

        Parameters
        ----------
        ability : float
            Person ability estimate on the logit scale.
        category : int
            Response category (0 to max_score).
        thresholds : array-like
            Uncentred threshold parameters, length equals max_score.

        Returns
        -------
        float
            Probability of the specified category, in [0, 1].
        """
        thresh = np.asarray(thresholds)
        m = len(thresh)
        cats = np.arange(m + 1, dtype=float)
        cumsum = np.concatenate(([0.0], np.cumsum(thresh)))
        log_nums = cats * ability - cumsum
        log_nums -= log_nums.max()
        nums = np.exp(log_nums)
        return nums[category] / nums.sum()

    def exp_score_uncentred(self, ability, thresholds):
        """
        Compute the expected score using uncentred threshold parameterisation.

        Calculates E[X | ability, thresholds] = sum(k * P(X=k)) over all
        categories, using uncentred threshold parameters.

        Parameters
        ----------
        ability : float
            Person ability estimate on the logit scale.
        thresholds : array-like
            Uncentred threshold parameters, length equals max_score.

        Returns
        -------
        float
            Expected score in [0, max_score].
        """
        thresh = np.asarray(thresholds)
        m = len(thresh)
        cats = np.arange(m + 1, dtype=float)
        cumsum = np.concatenate(([0.0], np.cumsum(thresh)))
        log_nums = cats * ability - cumsum
        log_nums -= log_nums.max()
        nums = np.exp(log_nums)
        return (cats * nums).sum() / nums.sum()

    def exp_score_centred(self, ability, difficulty, thresholds):
        """
        Compute the expected score using centred parameterisation.

        Calculates E[X | ability, difficulty, thresholds] using the centred
        PCM formulation with a central item difficulty and Rasch-Andrich offsets.
        Numerically stabilised via log-sum-exp.

        Parameters
        ----------
        ability : float
            Person ability estimate on the logit scale.
        difficulty : float
            Central item difficulty on the logit scale.
        thresholds : array-like
            Rasch-Andrich threshold offsets, length max_score + 1.

        Returns
        -------
        float
            Expected score in [0, max_score].
        """
        thresh = np.asarray(thresholds)
        max_score = len(thresh) - 1
        cats = np.arange(max_score + 1, dtype=float)
        cumsum = np.concatenate(([0.0], np.cumsum(thresh[1:])))
        log_nums = cats * (ability - difficulty) - cumsum
        log_nums -= log_nums.max()
        nums = np.exp(log_nums)
        return (cats * nums).sum() / nums.sum()

    def variance_uncentred(self, ability, thresholds):
        """
        Compute item variance (Fisher information) using uncentred parameterisation.

        Calculates Var[X | ability, thresholds] = sum((k - E[X])^2 * P(X=k)),
        equal to the Fisher information for the item at the given ability.
        Numerically stabilised via log-sum-exp.

        Parameters
        ----------
        ability : float
            Person ability estimate on the logit scale.
        thresholds : array-like
            Uncentred threshold parameters, length equals max_score.

        Returns
        -------
        float
            Item variance / Fisher information. Always non-negative.
        """
        thresh = np.asarray(thresholds)
        m = len(thresh)
        cats = np.arange(m + 1, dtype=float)
        cumsum = np.concatenate(([0.0], np.cumsum(thresh)))
        log_nums = cats * ability - cumsum
        log_nums -= log_nums.max()
        nums = np.exp(log_nums)
        probs = nums / nums.sum()
        expected = (cats * probs).sum()
        return ((cats - expected) ** 2 * probs).sum()

    def variance_centred(self, ability, difficulty, thresholds):
        """
        Compute item variance (Fisher information) using centred parameterisation.

        Calculates Var[X | ability, difficulty, thresholds] = sum((k - E[X])^2 * P(X=k)).
        Numerically stabilised via log-sum-exp.

        Parameters
        ----------
        ability : float
            Person ability estimate on the logit scale.
        difficulty : float
            Central item difficulty on the logit scale.
        thresholds : array-like
            Rasch-Andrich threshold offsets, length max_score + 1.

        Returns
        -------
        float
            Item variance / Fisher information.
        """
        thresh = np.asarray(thresholds)
        max_score = len(thresh) - 1
        cats = np.arange(max_score + 1, dtype=float)
        cumsum = np.concatenate(([0.0], np.cumsum(thresh[1:])))
        log_nums = cats * (ability - difficulty) - cumsum
        log_nums -= log_nums.max()
        nums = np.exp(log_nums)
        probs = nums / nums.sum()
        expected = (cats * probs).sum()
        return ((cats - expected) ** 2 * probs).sum()

    def kurtosis_uncentred(self, ability, thresholds):
        """
        Compute the fourth central moment of the response distribution (uncentred).

        Calculates sum((k - E[X])^4 * P(X=k)) using uncentred threshold
        parameterisation. Used in the Wilson-Hilferty approximation for
        standardised fit statistics (Infit Z, Outfit Z).

        Parameters
        ----------
        ability : float
            Person ability estimate on the logit scale.
        thresholds : array-like
            Uncentred threshold parameters, length equals max_score.

        Returns
        -------
        float
            Fourth central moment of the response distribution.
        """
        thresh = np.asarray(thresholds)
        m = len(thresh)
        cats = np.arange(m + 1, dtype=float)
        cumsum = np.concatenate(([0.0], np.cumsum(thresh)))
        log_nums = cats * ability - cumsum
        log_nums -= log_nums.max()
        nums = np.exp(log_nums)
        probs = nums / nums.sum()
        expected = (cats * probs).sum()
        return ((cats - expected) ** 4 * probs).sum()

    def kurtosis_centred(self, ability, difficulty, thresholds):
        """
        Compute the fourth central moment of the response distribution (centred).

        Calculates sum((k - E[X])^4 * P(X=k)) using centred PCM parameterisation.

        Parameters
        ----------
        ability : float
            Person ability estimate on the logit scale.
        difficulty : float
            Central item difficulty on the logit scale.
        thresholds : array-like
            Rasch-Andrich threshold offsets, length max_score + 1.

        Returns
        -------
        float
            Fourth central moment of the response distribution.
        """
        thresh = np.asarray(thresholds)
        max_score = len(thresh) - 1
        cats = np.arange(max_score + 1, dtype=float)
        cumsum = np.concatenate(([0.0], np.cumsum(thresh[1:])))
        log_nums = cats * (ability - difficulty) - cumsum
        log_nums -= log_nums.max()
        nums = np.exp(log_nums)
        probs = nums / nums.sum()
        expected = (cats * probs).sum()
        return ((cats - expected) ** 4 * probs).sum()

    # ------------------------------------------------------------------
    # Vectorised cat-probability engine (core internal workhorse)
    # ------------------------------------------------------------------

    def _cat_probs_matrix(self, abilities, items, thresholds=None):
        '''
        Vectorised category probability computation used by abil(), warm(),
        fit_statistics(), csem(), and abil_lookup_table().

        Replaces five copies of the nested  "for item / for cat / sum()"  loop.

        Returns
        -------
        probs : ndarray, shape (max_max_score+1, N, n_items)
            probs[cat, person_idx, item_idx] = P(X=cat | ability, item)
            Categories beyond an item's max_score are set to 0.
        cats_arr : ndarray, shape (max_max_score+1,)
        '''
        if thresholds is None:
            thresholds = self.thresholds_uncentred

        ab = np.asarray(abilities, dtype=float)   # (N,)
        n = len(ab)
        n_items = len(items)
        max_max_score = int(max(len(thresholds[it]) for it in items))
        cats_arr = np.arange(max_max_score + 1, dtype=float)  # (C,)

        log_probs = np.full((max_max_score + 1, n, n_items), -np.inf)

        for j, item in enumerate(items):
            thresh = np.asarray(thresholds[item], dtype=float)   # (m,)
            m = len(thresh)
            # prefix sums: cumsum[k] = sum(thresh[:k])
            cumsum = np.concatenate(([0.0], np.cumsum(thresh)))  # (m+1,)
            # log numerator for category k: k*ability - cumsum[k]
            # shape: (m+1, N)
            log_num = cats_arr[:m + 1, None] * ab[None, :] - cumsum[:, None]
            log_probs[:m + 1, :, j] = log_num

        # Numerically stable softmax along category axis
        log_max = np.max(log_probs, axis=0, keepdims=True)      # (1, N, n_items)
        with np.errstate(invalid='ignore'):
            probs = np.exp(log_probs - log_max)
        probs[~np.isfinite(log_probs)] = 0.0
        probs /= probs.sum(axis=0, keepdims=True)

        return probs, cats_arr

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate(self,
                  constant=0.1,
                  method='cos',
                  matrix_power=3,
                  log_lik_tol=0.000001):
        """
        Estimate item thresholds using the PAIR (Pairwise) algorithm.

        Constructs a joint score-category frequency matrix across all item
        pairs and threshold combinations using vectorised operations, then
        raises it to successive powers to resolve structural zeroes (Choppin's
        matrix power property). A priority vector is extracted from the resolved
        matrix to obtain uncentred threshold estimates. Central item difficulties
        are derived as the mean of each item's uncentred thresholds, and centred
        thresholds are computed as deviations from this mean.

        Issues a UserWarning if only one item is present, or if constant=0
        and any item has all-maximum scores.

        Parameters
        ----------
        constant : float, default 0.1
            Additive smoothing constant applied to the frequency matrix.
            Use 0 to disable smoothing; estimation may fail if any item
            has all-maximum or all-minimum scores.
        method : str, default 'cos'
            Priority vector extraction method. See base.priority_vector().
        matrix_power : int, default 3
            Initial matrix power before checking for structural zeroes.
        log_lik_tol : float, default 0.000001
            Log-likelihood convergence tolerance for priority vector extraction.

        Attributes set
        --------------
        thresholds_uncentred : dict
            {item: numpy.ndarray} of uncentred threshold estimates per item.
        central_diffs : pandas.Series
            Central item difficulty (mean of uncentred thresholds) per item.
        thresholds_centred : dict
            {item: numpy.ndarray} of centred threshold offsets per item.
        threshold_list : numpy.ndarray
            Flat array of all uncentred thresholds concatenated.
        null_persons : pandas.Index
            Persons dropped prior to calibration due to entirely missing data.
        """

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

        all_null_mask = self.dataframe.isnull().all(axis=1)
        self.null_persons = self.dataframe.index[all_null_mask]
        if all_null_mask.any():
            self.dataframe = self.dataframe.loc[~all_null_mask]
        self.no_of_persons = self.dataframe.shape[0]

        df_array = self.dataframe.to_numpy()
        cum_scores = np.concatenate(([0], np.cumsum(self.max_score_vector.to_numpy())))
        total_matrix_dim = cum_scores[-1]
        matrix = np.zeros((total_matrix_dim, total_matrix_dim), dtype=np.float64)

        for item_1 in range(self.no_of_items):
            max_k1 = self.max_score_vector.iloc[item_1]
            start_1 = cum_scores[item_1]

            for item_2 in range(self.no_of_items):
                max_k2 = self.max_score_vector.iloc[item_2]
                start_2 = cum_scores[item_2]

                s1 = df_array[:, item_1]
                s2 = df_array[:, item_2]
                valid_mask = ~np.isnan(s1) & ~np.isnan(s2)
                if not np.any(valid_mask):
                    continue

                s1_valid = s1[valid_mask].astype(int)
                s2_valid = s2[valid_mask].astype(int)

                for i in range(max_k1):
                    m1 = (s1_valid == i + 1)
                    if np.any(m1):
                        counts = np.bincount(s2_valid[m1], minlength=max_k2)[:max_k2]
                        matrix[start_1 + i, start_2:start_2 + max_k2] = counts

        constant_matrix = ((matrix + matrix.T) > 0).astype(np.float64) * constant
        matrix += constant_matrix
        np.fill_diagonal(matrix, matrix.diagonal() + constant)

        mat = np.linalg.matrix_power(matrix, matrix_power)
        mat_pow = matrix_power
        while 0 in mat:
            mat = mat @ matrix
            mat_pow += 1
            if mat_pow == matrix_power + 5:
                mat += constant
                break

        threshold_vector = self.priority_vector(mat, method=method,
                                                log_lik_tol=log_lik_tol, pcm=True)
        self.threshold_list = threshold_vector

        split_indices = cum_scores[1:-1]
        threshold_vector_np = (threshold_vector.to_numpy()
                               if hasattr(threshold_vector, 'to_numpy')
                               else np.array(threshold_vector))
        item_threshold_arrays = np.split(threshold_vector_np, split_indices)

        self.thresholds_uncentred = {}
        self.central_diffs = {}
        self.thresholds_centred = {}

        for i, item in enumerate(self.dataframe.columns):
            uncentred = item_threshold_arrays[i]
            item_mean = np.mean(uncentred)
            self.thresholds_uncentred[item] = pd.Series(uncentred)
            self.central_diffs[item] = item_mean
            centered = np.zeros(len(uncentred) + 1)
            centered[1:] = uncentred - item_mean
            self.thresholds_centred[item] = centered

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

        if not hasattr(self, 'central_diffs'):
            self.calibrate(constant=constant, method=method,
                           matrix_power=matrix_power, log_lik_tol=log_lik_tol)

        x = anchors.sort_index()
        y = self.central_diffs.copy()[x.index].sort_index()

        difference = x - y
        median_difference = np.median(difference)
        mad = np.median(abs(difference - median_difference))
        robust_z = 0.6745 * (difference - median_difference) / mad
        abs_z = abs(robust_z)

        drop_x, drop_y = {}, {}
        keep_x, keep_y = x.copy(), y.copy()

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

        while (sd_ratio > sd_ratio_tol or
               sd_ratio < 1 / sd_ratio_tol or
               correlation < correlation_tol):

            drop_item = abs_z.idxmax()
            drop_x[drop_item] = x[drop_item]
            drop_y[drop_item] = y[drop_item]
            keep_x = keep_x.drop(labels=drop_item)
            keep_y = keep_y.drop(labels=drop_item)
            abs_z = abs_z.drop(labels=drop_item)

            if len(abs_z) < min_anchors:
                fail = True
                break
            correlation = np.corrcoef(keep_x, keep_y)[0, 1]
            sd_ratio = np.std(keep_x) / np.std(keep_y)

        if fail:
            print('Anchoring failed: too few anchors. Please review data and parameters.')
            return

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

        super_min = min(x.min(), y.min())
        super_max = max(x.max(), y.max())
        offset = (super_max - super_min) / 40

        b, a = np.polyfit(keep_x, keep_y, deg=1)
        reg_line_points = np.linspace(super_min, super_max, 2)
        ax.plot(reg_line_points, a + b * reg_line_points, color='darkred')

        for txt, xi, yi in zip(keep_x.keys(), keep_x.values, keep_y.values):
            ax.annotate(txt, (xi + offset, yi - offset / 2))
        for txt, xi, yi in zip(drop_x.keys(), drop_x.values, drop_y.values):
            ax.annotate(txt, (xi + offset, yi - offset / 2))

        ax.set_xlabel('Anchor difficulty')
        ax.set_ylabel('Calibrated difficulty')
        ax.legend(['Used anchor item', 'Unused anchor item'])
        plt.savefig('anchor_selection.png', dpi=300)
        plt.close(fig)

        self.central_diffs_anchor = self.central_diffs.copy() + self.anchor_trans_constant
        for item in anchors.index:
            self.central_diffs_anchor[item] = anchors.loc[item]

        self.thresholds_uncentred_anchor = {
            item: pd.Series(
                self.thresholds_centred[item][1:] + self.central_diffs_anchor[item],
                index=self.thresholds_uncentred[item].index
            )
            for item in self.dataframe.columns
        }
        # BUG FIX: was anchors[0] (first element of Series) — should be anchors[item]
        for item in self.dataframe.columns:
            if item in anchors.index:
                self.thresholds_uncentred_anchor[item].iloc[0] = anchors[item]

    # ------------------------------------------------------------------
    # Standard errors (bootstrap)
    # ------------------------------------------------------------------

    def std_errors(self,
                   interval=None,
                   constant=0.1,
                   method='cos',
                   matrix_power=3,
                   log_lik_tol=0.000001,
                   no_of_samples=100):
        """
        Estimate bootstrap standard errors for item threshold estimates.

        Draws no_of_samples bootstrap resamples of person-level response data,
        calibrates each resample, and computes the standard deviation of
        threshold and central difficulty estimates across samples. Also
        computes category width SEs (SE of spacing between adjacent thresholds).

        Parameters
        ----------
        interval : float or None, default None
            Confidence interval width (e.g. 0.95). If None, only SEs computed.
        constant : float, default 0.1
            Additive smoothing constant for bootstrap calibrations.
        method : str, default 'cos'
            Priority vector extraction method.
        matrix_power : int, default 3
            Matrix power for bootstrap calibrations.
        log_lik_tol : float, default 0.000001
            Convergence tolerance for bootstrap calibrations.
        no_of_samples : int, default 100
            Number of bootstrap resamples.

        Attributes set
        --------------
        threshold_se : dict
            {item: numpy.ndarray} bootstrap SEs for each item's uncentred thresholds.
        item_se : pandas.Series
            Bootstrap SE for each item's central difficulty.
        cat_width_se : dict
            {item: numpy.ndarray} bootstrap SEs for category widths.
        threshold_low / threshold_high : dict or None
            Bootstrap CI bounds for thresholds, or None.
        central_bootstrap : pandas.DataFrame
            Bootstrap central difficulty estimates, shape (no_of_samples, items).
        threshold_bootstrap : dict
            {item: DataFrame} of bootstrap threshold estimates.
        """
        samples = [PCM(self.dataframe.sample(frac=1, replace=True),
                       self.max_score_vector)
                   for _ in range(no_of_samples)]

        for sample in samples:
            sample.calibrate(constant=constant, method=method,
                             matrix_power=matrix_power, log_lik_tol=log_lik_tol)

        self.bootstrap_sample_thresholds = {
            f'Sample_{i + 1}': sample.thresholds_uncentred
            for i, sample in enumerate(samples)
        }

        calibrations_thresholds = {
            item: np.stack([samples[s].thresholds_uncentred[item]
                            for s in range(no_of_samples)])
            for item in self.items
        }
        calibrations_central = {
            item: np.array([samples[s].central_diffs[item]
                            for s in range(no_of_samples)])
            for item in self.items
        }

        sample_index = [f'Sample {i + 1}' for i in range(no_of_samples)]

        self.central_bootstrap = pd.DataFrame(calibrations_central, index=sample_index)

        self.threshold_bootstrap = {}
        for item in self.items:
            df_b = pd.DataFrame(calibrations_thresholds[item], index=sample_index)
            df_b.columns = np.arange(1, df_b.shape[1] + 1)
            self.threshold_bootstrap[item] = df_b

        self.cat_width_bootstrap = {}
        for item in self.items:
            if self.max_score_vector[item] == 1:
                self.cat_width_bootstrap[item] = pd.DataFrame(
                    {1: pd.Series({f'Sample {i + 1}': 0 for i in range(no_of_samples)})}
                )
            else:
                tb = self.threshold_bootstrap[item]
                cwb = pd.DataFrame(index=sample_index)
                for score in range(self.max_score_vector[item] - 1):
                    cwb[score + 1] = tb[score + 2] - tb[score + 1]
                self.cat_width_bootstrap[item] = cwb

        self.threshold_se = {item: np.std(calibrations_thresholds[item], axis=0)
                             for item in self.dataframe.columns}
        self.cat_width_se = {item: np.std(self.cat_width_bootstrap[item], axis=0)
                             for item in self.items}
        self.item_se = pd.Series({item: np.std(calibrations_central[item])
                                     for item in self.dataframe.columns})

        if interval is not None:
            lo = (1 - interval) * 50
            hi = (1 + interval) * 50
            self.threshold_low = {item: np.percentile(calibrations_thresholds[item], lo, axis=0)
                                  for item in self.dataframe.columns}
            self.threshold_high = {item: np.percentile(calibrations_thresholds[item], hi, axis=0)
                                   for item in self.dataframe.columns}
            self.cat_width_low = {item: np.percentile(self.cat_width_bootstrap[item], lo, axis=0)
                                  for item in self.dataframe.columns}
            self.cat_width_high = {item: np.percentile(self.cat_width_bootstrap[item], hi, axis=0)
                                   for item in self.dataframe.columns}
            self.central_low = pd.Series({item: np.percentile(calibrations_central[item], lo)
                                          for item in self.dataframe.columns})
            self.central_high = pd.Series({item: np.percentile(calibrations_central[item], hi)
                                           for item in self.dataframe.columns})
        else:
            # BUG FIX: original shadowed threshold_low/high unconditionally,
            # wiping values set in the interval branch above.
            self.threshold_low = None
            self.threshold_high = None
            self.cat_width_low = None
            self.cat_width_high = None
            self.central_low = None
            self.central_high = None

    # ------------------------------------------------------------------
    # Ability estimation
    # ------------------------------------------------------------------

    def abil(self,
             persons,
             items=None,
             warm_corr=True,
             tolerance=0.00001,
             max_iters=100,
             ext_score_adjustment=0.5):
        """
        Estimate person abilities using Newton-Raphson maximum likelihood.

        For each person, iteratively solves the likelihood equation using
        uncentred threshold parameterisation and vectorised category probability
        computation. Extreme scores are adjusted. Optionally applies Warm (1989)
        bias correction.

        Parameters
        ----------
        persons : str or list
            Person identifier(s). Pass 'all' for all persons.
        items : str, list, or None, default None
            Item subset. None uses all items.
        warm_corr : bool, default True
            If True, applies Warm's (1989) bias correction.
        tolerance : float, default 0.00001
            Newton-Raphson convergence criterion.
        max_iters : int, default 100
            Maximum Newton-Raphson iterations.
        ext_score_adjustment : float, default 0.5
            Adjustment applied to extreme scores.

        Returns
        -------
        pandas.Series
            Ability estimates in logits. Returns numpy.nan on failure.
        """
        if isinstance(persons, str):
            persons = self.persons if persons == 'all' else [persons]

        if items is None:
            items = list(self.items)
        elif isinstance(items, str):
            items = list(self.items) if items in ('all', ) else [items]

        thresholds = {item: self.thresholds_uncentred[item] for item in items}
        person_data = self.dataframe.loc[persons, items]
        person_filter = person_data.notna().astype(float)

        scores = person_data.sum(axis=1).astype(float)
        ext_scores = person_filter.mul(self.max_score_vector[items], axis=1).sum(axis=1)

        scores = scores.clip(lower=ext_score_adjustment)
        scores[scores == ext_scores] -= ext_score_adjustment
        # Avoid modifying original if ext_scores match before clip adjustment
        scores = scores.where(scores < ext_scores, ext_scores - ext_score_adjustment)

        thresh_sums = pd.Series({item: thresholds[item].sum() for item in items})
        # Weighted mean threshold per person (accounting for missing items)
        thresh_sum_df = person_filter.mul(thresh_sums, axis=1)
        max_score_df = person_filter.mul(self.max_score_vector[items], axis=1)
        mean_diffs = thresh_sum_df.sum(axis=1) / max_score_df.sum(axis=1)

        try:
            estimates = np.log(scores) - np.log(ext_scores - scores) + mean_diffs
            items_list = list(items)

            # Per-person convergence mask. The original code accidentally used
            # nan propagation from exp() overflow to freeze converged persons
            # (abs(nan) > tol = False). Our log-sum-exp implementation gives
            # valid probabilities for all ability values, so we must track
            # convergence explicitly: once a person's change drops below
            # tolerance, exclude them from further updates. Without this,
            # slowly-converging persons keep updating everyone, and persons
            # near extreme scores accumulate drift of +/-1 logit per iteration
            # over max_iters steps, producing e.g. ability=117 logits.
            active = pd.Series(True, index=persons)
            iters = 0

            while active.any() and iters <= max_iters:
                active_persons = persons[active] if hasattr(persons, '__getitem__') else [p for p, a in zip(persons, active) if a]
                active_idx = active[active].index

                probs, cats_arr = self._cat_probs_matrix(
                    estimates.loc[active_idx].values, items_list, thresholds
                )
                # probs: (C, N_active, I)
                exp_score = (cats_arr[:, None, None] * probs).sum(axis=0)        # (N_active, I)
                exp_score_df = pd.DataFrame(exp_score, index=active_idx, columns=items_list)
                exp_score_df *= person_filter.loc[active_idx]

                dev = cats_arr[:, None, None] - exp_score[None, :, :]            # (C, N_active, I)
                info = (dev ** 2 * probs).sum(axis=0)                             # (N_active, I)
                info_df = pd.DataFrame(info, index=active_idx, columns=items_list)
                info_df *= person_filter.loc[active_idx]

                result_list = exp_score_df.sum(axis=1)
                info_list   = info_df.sum(axis=1)

                changes = ((result_list - scores.loc[active_idx]) / info_list).clip(-1, 1)
                estimates.loc[active_idx] -= changes

                # Freeze persons whose change is now within tolerance
                active.loc[active_idx] = abs(changes) > tolerance
                iters += 1

            if iters >= max_iters and active.any():
                n_nc = int(active.sum())
                print(f'Warning: {n_nc} person(s) did not converge and will be '
                      f'set to NaN. Check for near-extreme scorers or increase max_iters.')
                estimates[active] = np.nan

            if warm_corr:
                # Apply Warm correction only to persons with valid (finite) estimates
                valid = estimates.notna()
                if valid.any():
                    estimates[valid] += self.warm(
                        estimates[valid],
                        items_list,
                        person_filter.loc[estimates.index[valid]]
                    )

        except Exception as e:
            print(f'abil() failed: {e}')
            estimates = pd.Series(np.nan, index=persons)

        return estimates

    def person_abils(self,
                     items=None,
                     warm_corr=True,
                     tolerance=0.00001,
                     max_iters=100,
                     ext_score_adjustment=0.5):
        """
        Estimate abilities for all persons and store as an attribute.

        Wrapper around abil() that estimates abilities for all persons
        and stores the result as self.person_abilities.

        Parameters
        ----------
        items : str, list, or None, default None
            Item subset. None uses all items.
        warm_corr : bool, default True
            Warm bias correction.
        tolerance : float, default 0.00001
            Newton-Raphson convergence tolerance.
        max_iters : int, default 100
            Maximum iterations.
        ext_score_adjustment : float, default 0.5
            Extreme score adjustment.

        Attributes set
        --------------
        person_abilities : pandas.Series
            Ability estimates for all persons, in logits.
        """
        self.person_abilities = self.abil(
            self.persons, items=items, warm_corr=warm_corr, tolerance=tolerance,
            max_iters=max_iters, ext_score_adjustment=ext_score_adjustment
        )

    def score_abil(self,
                   score,
                   items=None,
                   warm_corr=True,
                   tolerance=0.00001,
                   max_iters=100,
                   ext_score_adjustment=0.5):
        """
        Convert a raw total score to an ability estimate via Newton-Raphson ML.

        Used internally to draw score lines on TCC plots.

        Parameters
        ----------
        score : int or float
            Raw total score. Extreme scores adjusted by ext_score_adjustment.
        items : list or None, default None
            Item subset. None uses all items.
        warm_corr : bool, default True
            Warm bias correction.
        tolerance : float, default 0.00001
            Convergence tolerance.
        max_iters : int, default 100
            Maximum iterations.
        ext_score_adjustment : float, default 0.5
            Adjustment for extreme scores.

        Returns
        -------
        float
            Ability estimate in logits.
        """
        # BUG FIX: original had a string-iteration bug when items was a single string item name.
        if items is None or (isinstance(items, str) and items in ('all', 'none')):
            items = list(self.items)
        elif isinstance(items, str):
            items = [items]

        thresholds = {item: self.thresholds_uncentred[item] for item in items}
        mean_diff = pd.concat(list(thresholds.values())).mean()
        ext_score = self.max_score_vector[items].sum()

        used_score = float(score)
        if used_score == 0:
            used_score = ext_score_adjustment
        elif used_score == ext_score:
            used_score -= ext_score_adjustment

        estimate = log(used_score) - log(ext_score - used_score) + mean_diff
        change = 1.0
        iters = 0

        while abs(change) > tolerance and iters <= max_iters:
            result = sum(self.exp_score_uncentred(estimate, thresholds[item])
                         for item in items)
            info = sum(self.variance_uncentred(estimate, thresholds[item])
                       for item in items)
            change = max(-1.0, min(1.0, (result - used_score) / info))
            estimate -= change
            iters += 1

        if warm_corr:
            person_filter = {item: True for item in items}
            estimate += self.warm(pd.Series({'score': estimate}),
                                  items,
                                  pd.DataFrame(person_filter, index=['score'])).iloc[0]

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

        if isinstance(items, str) and items in ('all', 'none'):
            items = None
        elif isinstance(items, str):
            items = [items]
        if items is None:
            items = list(self.items)

        thresholds = {item: self.thresholds_uncentred[item] for item in items}
        total_max = self.max_score_vector.loc[items].sum()

        if ext_scores:
            scores = np.arange(total_max + 1)
            used_scores = scores.astype(float)
            used_scores[0] += ext_score_adjustment
            used_scores[-1] -= ext_score_adjustment
        else:
            scores = np.arange(1, total_max)
            used_scores = scores.astype(float)

        mean_diff = pd.concat(list(thresholds.values())).mean()
        estimates = pd.Series(
            np.log(used_scores) - np.log(total_max - used_scores) + mean_diff,
            index=scores
        )

        changes = pd.Series(1.0, index=scores)
        iters = 0

        while abs(changes).max() > tolerance and iters <= max_iters:
            probs, cats_arr = self._cat_probs_matrix(estimates.values, items, thresholds)
            exp_score = (cats_arr[:, None, None] * probs).sum(axis=0)     # (N, I)
            exp_score_df = pd.DataFrame(exp_score, index=scores, columns=items)

            dev = cats_arr[:, None, None] - exp_score[None, :, :]
            info = (dev ** 2 * probs).sum(axis=0)
            info_df = pd.DataFrame(info, index=scores, columns=items)

            result_list = exp_score_df.sum(axis=1)
            info_list = info_df.sum(axis=1)

            changes = ((result_list - used_scores) / info_list).clip(-1, 1)
            estimates -= changes
            iters += 1

        if warm_corr:
            person_filter = pd.DataFrame(1.0, columns=items, index=scores)
            estimates += self.warm(estimates, items, person_filter)

        self.abil_table = estimates

    def warm(self, abilities, items, person_filter):
        """
        Apply Warm's (1989) weighted maximum likelihood bias correction.

        Uses the vectorised _cat_probs_matrix engine. Computes
        (J1 - J2 + J3) / (2 * I^2) simultaneously for all persons.

        Parameters
        ----------
        abilities : pandas.Series
            Current ability estimates.
        items : list or pandas.Index
            Item subset.
        person_filter : pandas.DataFrame
            Binary mask (1 = responded, NaN = missing).

        Returns
        -------
        pandas.Series
            Warm bias correction terms to add to ML estimates.
        """
        if isinstance(items, str):
            items = [items]
        items = list(items)
        thresholds = {item: self.thresholds_uncentred[item] for item in items}

        probs, cats_arr = self._cat_probs_matrix(abilities.values, items, thresholds)
        # probs: (C, N, I)

        pf = person_filter.values if isinstance(person_filter, pd.DataFrame) else None

        exp_score = (cats_arr[:, None, None] * probs).sum(axis=0)           # (N, I)
        if pf is not None:
            exp_score *= pf

        dev = cats_arr[:, None, None] - exp_score[None, :, :]               # (C, N, I)
        info = (dev ** 2 * probs).sum(axis=0)                                # (N, I)
        if pf is not None:
            info *= pf

        # Warm correction numerator components.
        # BUG FIX: part_1 must use person-filter-masked probabilities.
        # probs is computed for ALL items; for persons with missing responses,
        # unobserved items have pf=0. part_2 and part_3 correctly use exp_score
        # and info (both already masked), but part_1 was summing k^3*P(k) over
        # ALL items including unobserved ones. With 6 unobserved items out of 12,
        # this inflated part_1 by ~289 units while part_2/part_3 reflected only
        # the 6 observed items, producing Warm corrections of +207 logits instead
        # of the correct -0.63 -- the source of ability estimates of +269 logits.
        cats3 = (cats_arr ** 3)[:, None, None]
        masked_probs = probs * pf[None, :, :] if pf is not None else probs   # (C, N, I)
        part_1 = (cats3 * masked_probs).sum(axis=0).sum(axis=1)              # (N,)

        exp_sq = exp_score ** 2
        part_2 = 3 * ((info + exp_sq) * exp_score).sum(axis=1)              # (N,)
        part_3 = 2 * (exp_score ** 3).sum(axis=1)                           # (N,)

        info_sum = info.sum(axis=1)                                          # (N,)
        den = 2 * info_sum ** 2

        warm_correction = (part_1 - part_2 + part_3) / den
        return pd.Series(warm_correction, index=abilities.index)

    def csem(self,
             persons=None,
             abilities=None,
             items=None):
        """
        Compute the conditional standard error of measurement.

        CSEM = 1 / sqrt(I) where I is total Fisher information.

        Parameters
        ----------
        persons : list, str, or None, default None
            Person identifiers. Overrides abilities if provided.
        abilities : pandas.Series, float, list, or None, default None
            Ability estimates. If None, uses self.person_abilities.
        items : str, list, or None, default None
            Item subset. None uses all items.

        Returns
        -------
        pandas.Series
            CSEM values in logits.
        """
        if abilities is None:
            abilities = self.person_abilities
        if isinstance(abilities, (int, float)):
            abilities = pd.Series({'Ability': float(abilities)})
        if isinstance(abilities, list):
            abilities = pd.Series({f'Ability {a}': a for a in abilities})
        if persons is not None:
            abilities = self.person_abilities.loc[persons]

        persons = abilities.index

        # BUG FIX: original `else` branch set thresholds = self.thresholds_uncentred[items]
        # which returns a column slice (wrong type) for a list of items.
        if items is None or (isinstance(items, str) and items == 'all'):
            items = list(self.items)
        elif isinstance(items, str):
            items = [items]

        thresholds = {item: self.thresholds_uncentred[item] for item in items}

        person_data = self.dataframe.loc[persons, items]
        person_filter = person_data.notna().astype(float)

        probs, cats_arr = self._cat_probs_matrix(abilities.values, items, thresholds)
        exp_score = (cats_arr[:, None, None] * probs).sum(axis=0)
        exp_score_df = pd.DataFrame(exp_score, index=persons, columns=items)
        exp_score_df *= person_filter

        dev = cats_arr[:, None, None] - exp_score[None, :, :]
        info = (dev ** 2 * probs).sum(axis=0)
        info_df = pd.DataFrame(info, index=persons, columns=items)
        info_df *= person_filter

        return 1.0 / (info_df.sum(axis=1) ** 0.5)

    # ------------------------------------------------------------------
    # Descriptive / count methods
    # ------------------------------------------------------------------

    def category_counts_item(self, item):
        if item not in self.dataframe.columns:
            print('Invalid item name')
            return None
        counts = (self.dataframe[item]
                  .value_counts()
                  .reindex(range(self.max_score_vector[item] + 1), fill_value=0)
                  .astype(int))
        return counts

    def category_counts_df(self):
        # Build per-item counts reindexed to that item's valid score range only.
        # Items with different max scores will have NaN for categories above their
        # maximum when combined into a single DataFrame -- these should display as
        # blank cells, not 0, to avoid implying those categories exist for that item.
        cat_counts_dict = {
            item: self.dataframe[item]
                      .value_counts()
                      .reindex(range(self.max_score_vector[item] + 1), fill_value=0)
                      .astype(int)
            for item in self.dataframe.columns
        }
        category_counts_df = pd.DataFrame(cat_counts_dict).T
        category_counts_df.sort_index(axis=1, inplace=True)

        category_counts_df['Total'] = self.dataframe.count()
        category_counts_df['Missing'] = self.no_of_persons - category_counts_df['Total']
        category_counts_df.loc['Total'] = category_counts_df.sum()

        # Convert valid counts to int, then replace NaN (categories above item
        # max score) with '' so the output table shows a blank rather than 0.
        category_counts_df = (category_counts_df
                              .fillna(-1)
                              .astype(int)
                              .replace(-1, ''))
        self.category_counts = category_counts_df

    # ------------------------------------------------------------------
    # Fit statistics
    # ------------------------------------------------------------------

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
                       no_of_samples=100,
                       interval=None):

        if not hasattr(self, 'thresholds_uncentred'):
            self.calibrate(constant=constant, method=method)
        if se and not hasattr(self, 'threshold_se'):
            self.std_errors(interval=interval, no_of_samples=no_of_samples,
                            constant=constant, method=method)
        if not hasattr(self, 'person_abilities'):
            self.person_abils(warm_corr=warm_corr, tolerance=tolerance,
                              max_iters=max_iters,
                              ext_score_adjustment=ext_score_adjustment)
        if not se:
            test_stats = False

        df = self.dataframe.copy()
        scores = df.sum(axis=1)
        max_scores = df.notna().mul(self.max_score_vector, axis=1).sum(axis=1)
        df = df[(scores > 0) & (scores < max_scores)]
        missing_mask = df.notna().astype(float)
        abilities = self.person_abilities.loc[df.index]

        # Safety net: exclude persons with extreme ability estimates.
        # Diverged NR iterations (e.g. near-perfect scorers on sparse response
        # patterns) can produce finite but astronomically large estimates such as
        # +117 logits. These are set to NaN in abil() when non-convergence is
        # detected, but guard here as well against any that slip through.
        # |ability| > 20 logits is well beyond any plausible true parameter value
        # and would produce kurtosis/info^2 ~ 1e+60 in the outfit q-factor.
        abilities = abilities[abilities.abs() <= 20]
        df = df.loc[abilities.index]
        missing_mask = missing_mask.loc[abilities.index]

        item_count = df.notna().sum(axis=0)
        person_count = df.notna().sum(axis=1)

        items_list = list(self.items)
        probs, cats_arr = self._cat_probs_matrix(
            abilities.values, items_list, self.thresholds_uncentred
        )
        # probs: (C, N, I)

        # Store cat_prob_dict for downstream use (e.g. trim_cat_prob_dict)
        self.cat_prob_dict = {
            cat: pd.DataFrame(probs[cat], index=abilities.index, columns=self.items)
            for cat in range(probs.shape[0])
        }
        if trim_cat_prob_dict:
            for cat in self.cat_prob_dict:
                self.cat_prob_dict[cat] = self.cat_prob_dict[cat].loc[df.index]

        exp_score = (cats_arr[:, None, None] * probs).sum(axis=0)          # (N, I)
        self.exp_score_df = pd.DataFrame(exp_score, index=abilities.index, columns=self.items)
        self.exp_score_df *= missing_mask

        dev = cats_arr[:, None, None] - exp_score[None, :, :]              # (C, N, I)
        info = (dev ** 2 * probs).sum(axis=0)                               # (N, I)
        self.info_df = pd.DataFrame(info, index=abilities.index, columns=self.items)
        self.info_df *= missing_mask

        kurtosis = ((dev ** 4) * probs).sum(axis=0)                         # (N, I)
        self.kurtosis_df = pd.DataFrame(kurtosis, index=abilities.index, columns=self.items)
        self.kurtosis_df *= missing_mask

        # Cell-level guard: exclude person-item cells where one category has
        # near-certain probability. This follows WINSTEPS convention (p > 0.9999).
        #
        # Why this is needed: _cat_probs_matrix() uses the log-sum-exp trick,
        # which correctly produces a very small positive info (rather than NaN
        # from exp-overflow as in the original). But kurtosis / info^2 in the
        # outfit q-factor then explodes. The person-level extreme-score filter
        # above (scores > 0 & scores < max_scores) removes persons with no
        # information at all, but does not catch the case where a non-extreme
        # person has an extreme response on a single easy or hard item.
        #
        # The p-based threshold is preferable to an info-based one because:
        #   (a) it has a direct probabilistic interpretation independent of
        #       item max score (unlike info, which scales with max_score^2)
        #   (b) it matches documented WINSTEPS exclusion criterion
        #   (c) 0.9999 corresponds to ~9.5 logits above/below the item threshold,
        #       well outside the range where the cell carries useful information
        P_THRESHOLD = 0.9999
        max_cat_prob = pd.DataFrame(
            probs.max(axis=0),          # (N, I): max prob across categories
            index=abilities.index,
            columns=self.items
        )
        degenerate_mask = max_cat_prob > P_THRESHOLD
        self.info_df[degenerate_mask] = np.nan
        self.kurtosis_df[degenerate_mask] = np.nan
        self.exp_score_df[degenerate_mask] = np.nan

        self.residual_df = self.dataframe.reindex(df.index) - self.exp_score_df
        self.std_residual_df = self.residual_df / (self.info_df ** 0.5)

        # --- Item fit ---
        self.item_outfit_ms = (self.std_residual_df ** 2).mean()
        self.item_infit_ms = (self.residual_df ** 2).sum() / self.info_df.sum()

        item_outfit_q = (((self.kurtosis_df / (self.info_df ** 2))
                          / (item_count ** 2)).sum() - (1 / item_count)) ** 0.5
        self.item_outfit_zstd = (((self.item_outfit_ms ** (1/3)) - 1)
                                 * (3 / item_outfit_q) + (item_outfit_q / 3))

        item_infit_q = ((self.kurtosis_df - self.info_df ** 2).sum()
                        / (self.info_df.sum() ** 2)) ** 0.5
        self.item_infit_zstd = (((self.item_infit_ms ** (1/3)) - 1)
                                * (3 / item_infit_q) + (item_infit_q / 3))

        self.response_counts = self.dataframe.count(axis=0)
        self.item_facilities = self.dataframe.mean(axis=0) / self.max_score_vector

        (self.point_measure,
         self.exp_point_measure) = self.pt_meas(self.person_abilities,
                                                self.exp_score_df,
                                                self.info_df)

        # --- Threshold fit (dichotomised) ---
        dich_thresh = {}
        for item in self.dataframe.columns:
            dich_thresh[item] = {}
            for t in range(self.max_score_vector[item]):
                col = self.dataframe[item].where(
                    self.dataframe[item].isin([t, t + 1]), np.nan
                ) - t
                dich_thresh[item][t + 1] = col

        dich_thresh_exp = {item: {} for item in self.dataframe.columns}
        dich_thresh_var = {item: {} for item in self.dataframe.columns}
        dich_thresh_kur = {item: {} for item in self.dataframe.columns}
        dich_residuals = {item: {} for item in self.dataframe.columns}
        dich_std_residuals = {item: {} for item in self.dataframe.columns}

        dich_thresh_count = {
            item: {t + 1: dich_thresh[item][t + 1].count()
                   for t in range(self.max_score_vector[item])}
            for item in self.dataframe.columns
        }

        for item in self.dataframe.columns:
            thresh_val = self.thresholds_uncentred[item]
            for t in range(self.max_score_vector[item]):
                diff = thresh_val.iloc[t]
                mm = dich_thresh[item][t + 1].notna().astype(float).replace(0, np.nan)

                p = 1.0 / (1.0 + np.exp(diff - self.person_abilities))
                p_masked = p * mm

                dich_thresh_exp[item][t + 1] = p_masked
                var = p_masked * (1 - p_masked)
                dich_thresh_var[item][t + 1] = var

                # Kurtosis for binary item
                dich_thresh_kur[item][t + 1] = (
                    ((-p_masked) ** 4) * (1 - p_masked) +
                    ((1 - p_masked) ** 4) * p_masked
                ) * mm

                dich_residuals[item][t + 1] = dich_thresh[item][t + 1] - p_masked
                dich_std_residuals[item][t + 1] = (
                    dich_residuals[item][t + 1] / (var ** 0.5)
                )

        def _concat_series(nested_dict):
            '''Helper: flatten {item: {t: Series}} -> MultiIndex Series.'''
            return pd.concat(
                {item: pd.Series(nested_dict[item]) for item in self.dataframe.columns},
                keys=self.dataframe.columns
            )

        self.threshold_outfit_ms = _concat_series({
            item: {
                t + 1: ((dich_std_residuals[item][t + 1] ** 2).sum()
                        / dich_thresh_count[item][t + 1]
                        if dich_thresh_count[item][t + 1] > 0 else np.nan)
                for t in range(self.max_score_vector[item])
            }
            for item in self.dataframe.columns
        })

        self.threshold_infit_ms = _concat_series({
            item: {
                t + 1: ((dich_residuals[item][t + 1] ** 2).sum()
                        / dich_thresh_var[item][t + 1].sum()
                        if dich_thresh_var[item][t + 1].sum() > 0 else np.nan)
                for t in range(self.max_score_vector[item])
            }
            for item in self.dataframe.columns
        })

        threshold_outfit_q_raw = _concat_series({
            item: {
                t + 1: (((dich_thresh_kur[item][t + 1]
                          / (dich_thresh_var[item][t + 1] ** 2))
                         / (dich_thresh_count[item][t + 1] ** 2)).sum()
                        - (1 / dich_thresh_count[item][t + 1])
                        if dich_thresh_count[item][t + 1] > 0 else np.nan)
                for t in range(self.max_score_vector[item])
            }
            for item in self.dataframe.columns
        })
        threshold_outfit_q = threshold_outfit_q_raw ** 0.5
        self.threshold_outfit_zstd = (((self.threshold_outfit_ms ** (1/3)) - 1)
                                      * (3 / threshold_outfit_q)
                                      + (threshold_outfit_q / 3))

        threshold_infit_q = _concat_series({
            item: {
                t + 1: ((dich_thresh_kur[item][t + 1]
                         - dich_thresh_var[item][t + 1] ** 2).sum()
                        / (dich_thresh_var[item][t + 1].sum() ** 2)
                        if dich_thresh_var[item][t + 1].sum() > 0 else np.nan)
                for t in range(self.max_score_vector[item])
            }
            for item in self.dataframe.columns
        }) ** 0.5
        self.threshold_infit_zstd = (((self.threshold_infit_ms ** (1/3)) - 1)
                                     * (3 / threshold_infit_q)
                                     + (threshold_infit_q / 3))

        abil_deviation = self.person_abilities - self.person_abilities.mean()

        # Threshold point-measure correlations
        pm_num = _concat_series({
            item: {
                t + 1: ((dich_thresh[item][t + 1] - dich_thresh[item][t + 1].mean())
                        * abil_deviation).sum()
                for t in range(self.max_score_vector[item])
            }
            for item in self.dataframe.columns
        })
        pm_den = _concat_series({
            item: {
                t + 1: (((dich_thresh[item][t + 1] - dich_thresh[item][t + 1].mean()) ** 2).sum()
                        * (abil_deviation ** 2).sum()) ** 0.5
                for t in range(self.max_score_vector[item])
            }
            for item in self.dataframe.columns
        })
        self.threshold_point_measure = pm_num / pm_den

        exp_pm_dict = {
            item: {
                t + 1: (dich_thresh_exp[item][t + 1]
                        - dich_thresh_exp[item][t + 1].mean()
                        if dich_thresh_exp[item][t + 1].count() > 0 else np.nan)
                for t in range(self.max_score_vector[item])
            }
            for item in self.dataframe.columns
        }
        exp_pm_num = _concat_series({
            item: {t + 1: (exp_pm_dict[item][t + 1] * abil_deviation).sum()
                   for t in range(self.max_score_vector[item])}
            for item in self.dataframe.columns
        })
        exp_pm_den_raw = _concat_series({
            item: {
                t + 1: ((exp_pm_dict[item][t + 1] ** 2)
                        + dich_thresh_var[item][t + 1]).sum()
                for t in range(self.max_score_vector[item])
            }
            for item in self.dataframe.columns
        })
        exp_pm_den = (exp_pm_den_raw * (abil_deviation ** 2).sum()) ** 0.5
        self.threshold_exp_point_measure = exp_pm_num / exp_pm_den

        self.threshold_rmsr = _concat_series({
            item: {
                t + 1: ((dich_residuals[item][t + 1] ** 2).sum()
                        / dich_residuals[item][t + 1].count()
                        if dich_residuals[item][t + 1].count() > 0 else np.nan)
                for t in range(self.max_score_vector[item])
            }
            for item in self.dataframe.columns
        }) ** 0.5

        diff_num = _concat_series({
            item: {
                t + 1: ((self.person_abilities - self.thresholds_uncentred[item].iloc[t])
                        * dich_residuals[item][t + 1]).sum()
                for t in range(self.max_score_vector[item])
            }
            for item in self.dataframe.columns
        })
        diff_den = _concat_series({
            item: {
                t + 1: (dich_thresh_var[item][t + 1]
                        * (self.person_abilities
                           - self.thresholds_uncentred[item].iloc[t]) ** 2).sum()
                for t in range(self.max_score_vector[item])
            }
            for item in self.dataframe.columns
        })
        self.threshold_discrimination = 1 + diff_num / diff_den

        # --- Person fit ---
        self.csem_vector = 1.0 / (self.info_df.sum(axis=1) ** 0.5)
        self.rsem_vector = ((self.residual_df ** 2).sum(axis=1) ** 0.5) / self.info_df.sum(axis=1)

        self.person_outfit_ms = (self.std_residual_df ** 2).mean(axis=1)
        self.person_outfit_ms.name = 'Outfit MS'
        self.person_infit_ms = (self.residual_df ** 2).sum(axis=1) / self.info_df.sum(axis=1)
        self.person_infit_ms.name = 'Infit MS'

        base_df = self.kurtosis_df / (self.info_df ** 2)
        base_df = base_df.div(person_count ** 2, axis=0)
        person_outfit_q = (base_df.sum(axis=1) - 1 / person_count) ** 0.5
        self.person_outfit_zstd = (((self.person_outfit_ms ** (1/3)) - 1)
                                   * (3 / person_outfit_q)
                                   + (person_outfit_q / 3))
        self.person_outfit_zstd.name = 'Outfit Z'

        person_infit_q = ((self.kurtosis_df - self.info_df ** 2).sum(axis=1)
                          / (self.info_df.sum(axis=1) ** 2)) ** 0.5
        self.person_infit_zstd = (((self.person_infit_ms ** (1/3)) - 1)
                                  * (3 / person_infit_q)
                                  + (person_infit_q / 3))
        self.person_infit_zstd.name = 'Infit Z'

        # --- Test-level fit ---
        if test_stats:
            thresh_flat = np.concatenate(
                [self.thresholds_uncentred[item].values
                 for item in self.dataframe.columns]
            )
            se_flat = np.concatenate(
                [self.threshold_se[item]
                 for item in self.dataframe.columns]
            )
            self.threshold_list = thresh_flat
            self.threshold_se_list = se_flat

            self.isi_central = (self.central_diffs.var()
                                / (self.item_se ** 2).mean() - 1) ** 0.5
            self.item_strata = (4 * self.isi_central + 1) / 3
            self.item_reliability = (self.isi_central ** 2
                                     / (1 + self.isi_central ** 2))

            self.isi_thresholds = (thresh_flat.var()
                                   / (se_flat ** 2).mean() - 1) ** 0.5
            self.threshold_strata = (4 * self.isi_thresholds + 1) / 3
            self.threshold_reliability = (self.isi_thresholds ** 2
                                          / (1 + self.isi_thresholds ** 2))

            self.psi = ((np.var(self.person_abilities)
                         - (self.rsem_vector ** 2).mean()) ** 0.5
                        / (self.rsem_vector ** 2).mean() ** 0.5)
            self.person_strata = (4 * self.psi + 1) / 3
            self.person_reliability = self.psi ** 2 / (1 + self.psi ** 2)

    # ------------------------------------------------------------------
    # Residual correlation / PCA
    # ------------------------------------------------------------------

    def res_corr_analysis(self,
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
        """
        Analyse standardised residual correlations for local item dependence.

        Computes inter-item standardised residual correlations and performs
        PCA to detect violations of local independence and unidimensionality.
        A first eigenvalue > 2.0 conventionally suggests a second dimension.
        Auto-triggers fit_statistics() if not yet run.

        Parameters
        ----------
        warm_corr : bool, default True
            Warm bias correction.
        tolerance : float, default 0.00001
            Newton-Raphson convergence tolerance.
        max_iters : int, default 100
            Maximum iterations.
        ext_score_adjustment : float, default 0.5
            Extreme score adjustment.
        constant : float, default 0.1
            Additive smoothing constant.
        method : str, default 'cos'
            Priority vector extraction method.
        matrix_power : int, default 3
            Matrix power for calibration.
        log_lik_tol : float, default 0.000001
            Convergence tolerance for calibration.
        no_of_samples : int, default 100
            Bootstrap samples.
        interval : float or None, default None
            CI width.

        Attributes set
        --------------
        residual_correlations : pandas.DataFrame
            Item-by-item correlation matrix of standardised residuals.
        eigenvectors, eigenvalues, variance_explained, loadings : DataFrame or None
            PCA results. None if PCA fails.
        pca_fail : bool
            True only if PCA raises an exception.
        """
        if not hasattr(self, 'std_residual_df'):
            self.fit_statistics(warm_corr=warm_corr, tolerance=tolerance,
                                max_iters=max_iters,
                                ext_score_adjustment=ext_score_adjustment,
                                constant=constant, method=method,
                                matrix_power=matrix_power,
                                log_lik_tol=log_lik_tol,
                                no_of_samples=no_of_samples,
                                interval=interval)

        self.residual_correlations = self.residual_df.corr(numeric_only=False)
        pca = PCA()
        try:
            pca.fit(self.std_residual_df.corr())
            n = self.no_of_items
            pc_labels = [f'PC {pc + 1}' for pc in range(n)]

            self.eigenvectors = pd.DataFrame(
                pca.components_,
                columns=[f'Eigenvector {pc + 1}' for pc in range(n)]
            )
            self.eigenvalues = pd.DataFrame(
                pca.explained_variance_, index=pc_labels, columns=['Eigenvalue']
            )
            self.variance_explained = pd.DataFrame(
                pca.explained_variance_ratio_,
                index=pc_labels, columns=['Variance explained']
            )
            self.loadings = pd.DataFrame(
                self.eigenvectors.values.T * (pca.explained_variance_ ** 0.5),
                index=self.dataframe.columns, columns=pc_labels
            )
        except Exception:
            self.pca_fail = True
            print('PCA of residuals failed.')
            self.eigenvectors = None
            self.eigenvalues = None
            self.variance_explained = None
            self.loadings = None

    # ------------------------------------------------------------------
    # Output tables
    # ------------------------------------------------------------------

    def item_stats_df(self,
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
                      no_of_samples=100,
                      interval=None):

        if full:
            zstd = True
            point_measure_corr = True
            if interval is None:
                interval = 0.95

        if (not hasattr(self, 'threshold_low') or
                self.threshold_low is None and interval is not None):
            self.std_errors(interval=interval, no_of_samples=no_of_samples,
                            constant=constant, method=method)
        if not hasattr(self, 'item_infit_ms'):
            self.fit_statistics(warm_corr=warm_corr, tolerance=tolerance,
                                max_iters=max_iters,
                                ext_score_adjustment=ext_score_adjustment,
                                method=method, constant=constant,
                                no_of_samples=no_of_samples, interval=interval)

        stats = pd.DataFrame(index=self.items)
        stats['Estimate'] = self.central_diffs.round(dp)
        stats['SE'] = self.item_se.round(dp)
        if interval is not None:
            stats[f'{round((1 - interval) * 50, 1)}%'] = self.central_low.round(dp)
            stats[f'{round((1 + interval) * 50, 1)}%'] = self.central_high.round(dp)
        stats['Count'] = self.response_counts.astype(int)
        stats['Facility'] = self.item_facilities.round(dp)
        stats['Infit MS'] = self.item_infit_ms.round(dp)
        if zstd:
            stats['Infit Z'] = self.item_infit_zstd.round(dp)
        stats['Outfit MS'] = self.item_outfit_ms.round(dp)
        if zstd:
            stats['Outfit Z'] = self.item_outfit_zstd.round(dp)
        if point_measure_corr:
            stats['PM corr'] = self.point_measure.round(dp)
            stats['Exp PM corr'] = self.exp_point_measure.round(dp)
        self.item_stats = stats

    def threshold_stats_df(self,
                           full=False,
                           zstd=True,
                           disc=False,
                           point_measure_corr=False,
                           dp=3,
                           warm_corr=True,
                           tolerance=0.00001,
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

        if not hasattr(self, 'threshold_infit_ms'):
            self.fit_statistics(warm_corr=warm_corr, tolerance=tolerance,
                                max_iters=max_iters,
                                ext_score_adjustment=ext_score_adjustment,
                                method=method, constant=constant,
                                no_of_samples=no_of_samples, interval=interval)

        estimate_array = np.concatenate(
            [self.thresholds_uncentred[item].values
             for item in self.dataframe.columns]
        )
        se_array = np.concatenate(
            [self.threshold_se[item] for item in self.dataframe.columns]
        )

        stats = pd.DataFrame(index=self.threshold_infit_ms.index)
        stats['Estimate'] = estimate_array.round(dp)
        stats['SE'] = se_array.round(dp)

        if interval is not None:
            low_array = np.concatenate(
                [self.threshold_low[item] for item in self.dataframe.columns]
            )
            high_array = np.concatenate(
                [self.threshold_high[item] for item in self.dataframe.columns]
            )
            stats[f'{round((1 - interval) * 50, 1)}%'] = low_array.round(dp)
            stats[f'{round((1 + interval) * 50, 1)}%'] = high_array.round(dp)

        stats['Infit MS'] = self.threshold_infit_ms.reset_index(drop=True).round(dp).values
        if zstd:
            stats['Infit Z'] = self.threshold_infit_zstd.reset_index(drop=True).round(dp).values
        stats['Outfit MS'] = self.threshold_outfit_ms.reset_index(drop=True).round(dp).values
        if zstd:
            stats['Outfit Z'] = self.threshold_outfit_zstd.reset_index(drop=True).round(dp).values
        if disc:
            stats['Discrim'] = self.threshold_discrimination.reset_index(drop=True).round(dp).values
        if point_measure_corr:
            stats['PM corr'] = self.threshold_point_measure.reset_index(drop=True).round(dp).values
            stats['Exp PM corr'] = self.threshold_exp_point_measure.reset_index(drop=True).round(dp).values

        self.threshold_stats_uncentred = stats

        central_array = np.concatenate(
            [np.full(self.max_score_vector[item], self.central_diffs[item])
             for item in self.dataframe.columns]
        )
        self.threshold_stats_centred = stats.copy()
        self.threshold_stats_centred['Estimate'] -= central_array.round(dp)
        if interval is not None:
            self.threshold_stats_centred[f'{round((1 - interval) * 50, 1)}%'] -= central_array.round(dp)
            self.threshold_stats_centred[f'{round((1 + interval) * 50, 1)}%'] -= central_array.round(dp)

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
        """
        Build and store the person statistics summary table.

        Auto-triggers fit_statistics() if not yet run. One row per person
        with ability estimate, CSEM, raw score, max score, proportion correct,
        and Infit/Outfit MS and Z statistics.

        Parameters
        ----------
        full : bool, default False
            If True, sets rsem=True.
        rsem : bool, default False
            If True, includes Residual SEM (RSEM) column.
        dp : int, default 3
            Decimal places.
        warm_corr : bool, default True
            Warm bias correction.
        tolerance : float, default 0.00001
            Newton-Raphson convergence tolerance.
        max_iters : int, default 100
            Maximum Newton-Raphson iterations.
        ext_score_adjustment : float, default 0.5
            Extreme score adjustment.
        method : str, default 'cos'
            Priority vector extraction method.
        constant : float, default 0.1
            Additive smoothing constant.

        Attributes set
        --------------
        person_stats : pandas.DataFrame
            Person statistics with persons as rows. Contains 'Estimate',
            'CSEM', 'Score', 'Max score', 'p', 'Infit MS', 'Infit Z',
            'Outfit MS', 'Outfit Z'. Optional: 'RSEM'.
        """
        if not hasattr(self, 'person_infit_ms'):
            self.fit_statistics(warm_corr=warm_corr, tolerance=tolerance,
                                max_iters=max_iters,
                                ext_score_adjustment=ext_score_adjustment,
                                method=method, constant=constant)
        if full:
            rsem = True

        idx = self.dataframe.index
        stats = pd.DataFrame(index=idx)
        stats['Estimate'] = self.person_abilities.round(dp)
        stats['CSEM'] = self.csem_vector.round(dp)
        if rsem:
            stats['RSEM'] = self.rsem_vector.round(dp)
        stats['Score'] = self.dataframe.sum(axis=1).astype(int)
        stats['Max score'] = (self.dataframe.notna().astype(int)
                              .mul(self.max_score_vector, axis=1)
                              .sum(axis=1).astype(int))
        stats['p'] = (stats['Score'] / stats['Max score']).round(dp)

        # BUG FIX: original used .update(dict) which ignores index alignment.
        # Direct assignment aligns on index correctly.
        for col, src in [('Infit MS', self.person_infit_ms),
                         ('Infit Z', self.person_infit_zstd),
                         ('Outfit MS', self.person_outfit_ms),
                         ('Outfit Z', self.person_outfit_zstd)]:
            stats[col] = np.nan
            stats.loc[src.index, col] = src.round(dp).values

        self.person_stats = stats

    def test_stats_df(self,
                      dp=3,
                      warm_corr=True,
                      tolerance=0.00001,
                      max_iters=100,
                      ext_score_adjustment=0.5,
                      method='cos',
                      constant=0.1):

        if not hasattr(self, 'psi'):
            self.fit_statistics(warm_corr=warm_corr, tolerance=tolerance,
                                max_iters=max_iters,
                                ext_score_adjustment=ext_score_adjustment,
                                method=method, constant=constant)

        self.test_stats = pd.DataFrame({
            'Items':      [self.central_diffs.mean(), self.central_diffs.std(),
                           self.isi_central, self.item_strata, self.item_reliability],
            'Thresholds': [self.threshold_list.mean(), self.threshold_list.std(),
                           self.isi_thresholds, self.threshold_strata,
                           self.threshold_reliability],
            'Persons':    [self.person_abilities.mean(), self.person_abilities.std(),
                           self.psi, self.person_strata, self.person_reliability],
        }, index=['Mean', 'SD', 'Separation ratio', 'Strata', 'Reliability'])
        self.test_stats = self.test_stats.round(dp)

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
                   no_of_samples=100,
                   interval=None):

        if not hasattr(self, 'item_stats'):
            self.item_stats_df(dp=dp, warm_corr=warm_corr, tolerance=tolerance,
                               max_iters=max_iters,
                               ext_score_adjustment=ext_score_adjustment,
                               method=method, constant=constant,
                               no_of_samples=no_of_samples, interval=interval)
        if not hasattr(self, 'threshold_stats_uncentred'):
            self.threshold_stats_df(dp=dp, warm_corr=warm_corr, tolerance=tolerance,
                                    max_iters=max_iters,
                                    ext_score_adjustment=ext_score_adjustment,
                                    method=method, constant=constant,
                                    no_of_samples=no_of_samples, interval=interval)
        if not hasattr(self, 'person_stats'):
            self.person_stats_df(dp=dp, warm_corr=warm_corr, tolerance=tolerance,
                                 max_iters=max_iters,
                                 ext_score_adjustment=ext_score_adjustment,
                                 method=method, constant=constant)
        if not hasattr(self, 'test_stats'):
            self.test_stats_df(dp=dp, warm_corr=warm_corr, tolerance=tolerance,
                               max_iters=max_iters,
                               ext_score_adjustment=ext_score_adjustment,
                               method=method, constant=constant)

        if format == 'xlsx':
            if not filename.endswith('.xlsx'):
                filename += '.xlsx'
            with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:  # BUG FIX: writer.save() deprecated
                self.item_stats.to_excel(writer, sheet_name='Item statistics')
                self.threshold_stats_uncentred.to_excel(writer, sheet_name='Threshold statistics uncentred')
                self.threshold_stats_centred.to_excel(writer, sheet_name='Threshold statistics centred')
                self.person_stats.to_excel(writer, sheet_name='Person statistics')
                self.test_stats.to_excel(writer, sheet_name='Test statistics')
        else:
            if filename.endswith('.csv'):
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
                       tolerance=0.00001,
                       max_iters=100,
                       ext_score_adjustment=0.5,
                       method='cos',
                       constant=0.1):

        if not hasattr(self, 'eigenvectors'):
            self.fit_statistics(warm_corr=warm_corr, tolerance=tolerance,
                                max_iters=max_iters,
                                ext_score_adjustment=ext_score_adjustment,
                                method=method, constant=constant)

        frames = [self.eigenvectors, self.eigenvalues,
                  self.variance_explained, self.loadings]
        sheet_names_single = 'Item residual analysis'
        sheet_names_multi = ['Eigenvectors', 'Eigenvalues',
                             'Variance explained', 'Principal Component loadings']
        csv_suffixes = ['_eigenvectors', '_eigenvalues',
                        '_variance_explained', '_principal_component_loadings']

        if format == 'xlsx':
            if not filename.endswith('.xlsx'):
                filename += '.xlsx'
            with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:  # BUG FIX: writer.save()
                if single:
                    row = 0
                    for frame in frames:
                        frame.round(dp).to_excel(writer,
                                                  sheet_name=sheet_names_single,
                                                  startrow=row, startcol=0)
                        row += frame.shape[0] + 2
                else:
                    for frame, sheet in zip(frames, sheet_names_multi):
                        frame.round(dp).to_excel(writer, sheet_name=sheet)
        else:
            if single:
                if not filename.endswith('.csv'):
                    filename += '.csv'
                with open(filename, 'a') as f:
                    for frame in frames:
                        frame.round(dp).to_csv(f)
                        f.write('\n')
            else:
                if filename.endswith('.csv'):
                    filename = filename[:-4]
                for frame, suffix in zip(frames, csv_suffixes):
                    frame.round(dp).to_csv(f'{filename}{suffix}.csv')

    # ------------------------------------------------------------------
    # Class intervals (for ICC/CRC observed data overlay)
    # ------------------------------------------------------------------

    def class_intervals(self, abilities, items=None, no_of_classes=5):
        class_groups = [f'class_{i + 1}' for i in range(no_of_classes)]
        if items is None:
            items = self.dataframe.columns.tolist()
        df = self.dataframe[items].dropna(how='any')
        abils = abilities.loc[df.index]

        quantiles = abils.quantile([(i + 1) / no_of_classes
                                    for i in range(no_of_classes - 1)])
        mask_dict = {
            'class_1': abils < quantiles.values[0],
            f'class_{no_of_classes}': abils >= quantiles.values[-1],
            **{f'class_{i + 2}': ((abils >= quantiles.values[i]) &
                                   (abils < quantiles.values[i + 1]))
               for i in range(no_of_classes - 2)}
        }
        mean_abilities = pd.Series({cg: abils[mask_dict[cg]].mean()
                                    for cg in class_groups})
        obs = pd.concat({cg: pd.Series(df[mask_dict[cg]].mean().sum())
                         for cg in class_groups})
        return mean_abilities, obs

    def class_intervals_cats(self, item, no_of_classes=5):
        class_groups = [f'class_{i + 1}' for i in range(no_of_classes)]
        df = self.dataframe[item].dropna()
        abils = self.person_abilities[df.index]
        quantiles = abils.quantile([(i + 1) / no_of_classes
                                    for i in range(no_of_classes - 1)])
        mask_dict = {
            'class_1': abils < quantiles.values[0],
            f'class_{no_of_classes}': abils >= quantiles.values[-1],
            **{f'class_{i + 2}': ((abils >= quantiles.values[i]) &
                                   (abils < quantiles.values[i + 1]))
               for i in range(no_of_classes - 2)}
        }
        mean_abilities = pd.Series({cg: abils[mask_dict[cg]].mean()
                                    for cg in class_groups})
        obs_props = np.array([
            [(df[mask_dict[cg]] == cat).sum() for cat in range(self.max_score_vector[item] + 1)]
            for cg in class_groups
        ], dtype=float)
        obs_props /= obs_props.sum(axis=1, keepdims=True)
        return mean_abilities, obs_props

    def class_intervals_thresholds(self, item, no_of_classes=5):
        if not hasattr(self, 'person_abilities'):
            self.person_abils(warm_corr=False)

        class_groups = [f'class_{i + 1}' for i in range(no_of_classes)]
        df = self.dataframe[item]
        abils = self.person_abilities

        def make_masks(abils_subset):
            q = abils_subset.quantile([(i + 1) / no_of_classes
                                       for i in range(no_of_classes - 1)])
            return {
                'class_1': abils_subset < q.values[0],
                f'class_{no_of_classes}': abils_subset >= q.values[-1],
                **{f'class_{i + 2}': ((abils_subset >= q.values[i]) &
                                       (abils_subset < q.values[i + 1]))
                   for i in range(no_of_classes - 2)}
            }

        mean_abilities, obs_props = [], []
        for t in range(self.max_score_vector[item]):
            mask = df.isin([t, t + 1])
            cond_df = df[mask]
            adj_abils = abils[mask]
            masks = make_masks(adj_abils)
            combined = pd.DataFrame({'abil': adj_abils, 'score': cond_df})
            mean_abilities.append([combined['abil'][masks[cg]].mean() for cg in class_groups])
            obs_props.append([(combined['score'][masks[cg]] - t).mean() for cg in class_groups])

        return np.array(mean_abilities).T, np.array(obs_props).T

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    def plot_data(self,
                  x_data,
                  y_data,
                  x_min=-5,
                  x_max=5,
                  y_max=0,
                  items=None,
                  obs=None,
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
        """
        Core plotting engine for all PCM item and test characteristic curves.

        Renders one or more curves against an ability x-axis with optional
        observed-data overlays, threshold lines, central difference lines,
        score lines, information lines, and CSEM lines. Called internally
        by icc(), crcs(), threshold_ccs(), iic(), tcc(), test_info(), and
        test_csem(). Not normally called directly by users.

        Parameters
        ----------
        x_data : array-like
            X-axis values (typically ability grid from -20 to 20).
        y_data : numpy.ndarray
            2-D array shape (len(x_data), n_curves).
        x_min : float, default -5
            Left x-axis limit.
        x_max : float, default 5
            Right x-axis limit.
        y_max : float, default 0
            Upper y-axis limit. If <= 0, auto-scaled to 110% of peak.
        items : str, list, or None
            Item(s) being plotted, for threshold/score line lookups.
        obs : bool, list, or None
            Controls observed data overlay.
        x_obs_data, y_obs_data : array-like
            Observed data point coordinates.
        thresh_lines : bool, default False
            Draw vertical lines at each uncentred threshold.
        central_diff : bool, default False
            Draw a vertical line at the item's central difficulty.
        score_lines_item : list, default [None, None]
            [item_name, list_of_scores] for item-level score lines.
        score_lines_test : list or None
            Raw total scores for test-level score reference lines.
        point_info_lines_item : list, default [None, None]
            [item_name, list_of_abilities] for item-level information lines.
        point_info_lines_test : list or None
            Abilities for test-level information reference lines.
        point_csem_lines : list or None
            Abilities for CSEM reference lines.
        score_labels : bool, default False
            Annotate score/CSEM line intersections with values.
        warm : bool, default True
            Unused; passed for API consistency.
        cat_highlight : int or None
            Category index to shade blue on the plot.
        graph_title : str, default ''
            Plot title.
        y_label : str, default ''
            Y-axis label.
        plot_style : str, default 'white'
            'white' or 'dark'.
        palette : str, default 'dark blue'
            Colour palette name.
        black : bool, default False
            If True, all curves are black.
        figsize : tuple, default (8, 6)
            Figure size in inches.
        font : str, default 'Times New Roman'
            Font family.
        title_font_size : int, default 15
            Title font size.
        axis_font_size : int, default 12
            Axis label font size.
        labelsize : int, default 12
            Tick label font size.
        tex : bool, default True
            Attempt LaTeX rendering.
        plot_density : int, default 300
            Output DPI when saving.
        filename : str or None
            If provided, saves the plot.
        file_format : str, default 'png'
            File format for saved plots.

        Returns
        -------
        matplotlib.figure.Figure
            The rendered Figure object.
        """
        if plot_style == 'dark':
            sns.set_style('darkgrid')
        else:
            sns.set_style('whitegrid')

        palette_dict = {
            'dark blue':   ['dark', 'royalblue'],
            'light blue':  ['light', 'cornflowerblue'],
            'dark red':    ['dark', 'firebrick'],
            'light red':   ['light', 'indianred'],
            'dark green':  ['dark', 'forestgreen'],
            'light green': ['light', 'mediumseagreen'],
            'dark grey':   ['dark', 'dimgrey'],
            'light grey':  ['light', 'darkgrey'],
            'dark multi':  ['dark', 'dark'],
            'light multi': ['light', 'muted'],
        }

        shade, base_color = palette_dict[palette]
        if shade == 'dark':
            color_map = (sns.color_palette('dark', as_cmap=True)
                         if palette == 'dark multi'
                         else sns.dark_palette(base_color, reverse=True, as_cmap=True))
        else:
            color_map = (sns.color_palette('muted', as_cmap=True)
                         if palette == 'light multi'
                         else sns.light_palette(base_color, reverse=True, as_cmap=True))

        with plt.rc_context({'font.family': font, 'font.size': axis_font_size}):
            graph, ax = plt.subplots(figsize=figsize)
            no_of_plots = y_data.shape[1]
            cNorm = colors.Normalize(vmin=0, vmax=no_of_plots + 2)

            if 'multi' not in palette:
                scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=color_map)

            for i in range(no_of_plots):
                col = ('black' if black
                       else (scalarMap.to_rgba(i) if 'multi' not in palette
                             else color_map[i]))
                ax.plot(x_data, y_data[:, i], '', color=col, label=i + 1)

            if obs is not None:
                try:
                    no_obs_cats = y_obs_data.shape[1]
                    x_is_series = isinstance(x_obs_data, pd.Series)
                    for j in range(no_obs_cats):
                        col = (scalarMap.to_rgba(j) if 'multi' not in palette
                               else color_map[j])
                        xd = x_obs_data if x_is_series else x_obs_data[:, j]
                        ax.plot(xd, y_obs_data[:, j], 'o', color=col)
                except Exception:
                    pass

            if items is not None:
                if isinstance(items, str) and items not in ('all', 'none'):
                    thresholds = {items: self.thresholds_uncentred[items]}
                elif isinstance(items, str):
                    thresholds = self.thresholds_uncentred
                else:
                    thresholds = {item: self.thresholds_uncentred[item] for item in items}
            else:
                thresholds = self.thresholds_uncentred

            if thresh_lines and isinstance(items, str) and items not in ('all', 'none', None):
                for thr in self.thresholds_uncentred[items]:
                    ax.axvline(x=thr, color='black', linestyle='--')

            if items is not None and central_diff:
                item_key = items if isinstance(items, str) else None
                if item_key and item_key not in ('all', 'none'):
                    ax.axvline(x=np.mean(list(thresholds[item_key])),
                               color='darkred', linestyle='--')

            if score_lines_item[1] is not None:
                item = score_lines_item[0]
                valid = (all(s > 0 for s in score_lines_item[1]) and
                         all(s < self.max_score_vector[item] for s in score_lines_item[1]))
                if valid:
                    for s in score_lines_item[1]:
                        abil = self.score_abil(s, items=list(thresholds.keys()), warm_corr=False)
                        ax.vlines(x=abil, ymin=-100, ymax=s, color='black', linestyles='dashed')
                        ax.hlines(y=s, xmin=-100, xmax=abil, color='black', linestyles='dashed')
                        if score_labels:
                            ax.text(abil + (x_max - x_min) / 100, y_max / 50, str(round(abil, 2)))
                            ax.text(x_min + (x_max - x_min) / 100, s + y_max / 50, str(s))
                else:
                    print('Invalid score for score line.')

            if score_lines_test is not None:
                max_score = (sum(self.max_score_vector) if items is None
                             else sum(self.max_score_vector[items]))
                if (all(s > 0 for s in score_lines_test) and
                        all(s < max_score for s in score_lines_test)):
                    item_keys = (self.dataframe.columns if items is None else items)
                    for s in score_lines_test:
                        abil = self.score_abil(s, items=list(item_keys), warm_corr=False)
                        ax.vlines(x=abil, ymin=-100, ymax=s, color='black', linestyles='dashed')
                        ax.hlines(y=s, xmin=-100, xmax=abil, color='black', linestyles='dashed')
                        if score_labels:
                            ax.text(abil + (x_max - x_min) / 100, y_max / 50, str(round(abil, 2)))
                            ax.text(x_min + (x_max - x_min) / 100, s + y_max / 50, str(s))
                else:
                    print('Invalid score for score line.')

            if point_info_lines_item[1] is not None:
                item = point_info_lines_item[0]
                for abil in point_info_lines_item[1]:
                    info = self.variance_uncentred(abil, self.thresholds_uncentred[item])
                    ax.vlines(x=abil, ymin=-100, ymax=info, color='black', linestyles='dashed')
                    ax.hlines(y=info, xmin=-100, xmax=abil, color='black', linestyles='dashed')
                    if score_labels:
                        ax.text(abil + (x_max - x_min) / 100, y_max / 50, str(round(abil, 2)))
                        ax.text(x_min + (x_max - x_min) / 100, info + y_max / 50, str(round(info, 3)))

            if point_info_lines_test is not None:
                item_keys = (self.dataframe.columns if items is None else items)
                for abil in point_info_lines_test:
                    info = sum(self.variance_uncentred(abil, self.thresholds_uncentred[it])
                               for it in item_keys)
                    ax.vlines(x=abil, ymin=-100, ymax=info, color='black', linestyles='dashed')
                    ax.hlines(y=info, xmin=-100, xmax=abil, color='black', linestyles='dashed')
                    if score_labels:
                        ax.text(abil + (x_max - x_min) / 100, y_max / 50, str(round(abil, 2)))
                        ax.text(x_min + (x_max - x_min) / 100, info + y_max / 50, str(round(info, 3)))

            if point_csem_lines is not None:
                item_keys = (self.dataframe.columns if items is None else items)
                for abil in point_csem_lines:
                    info = sum(self.variance_uncentred(abil, self.thresholds_uncentred[it])
                               for it in item_keys)
                    csem = 1.0 / (info ** 0.5)
                    ax.vlines(x=abil, ymin=-100, ymax=csem, color='black', linestyles='dashed')
                    ax.hlines(y=csem, xmin=-100, xmax=abil, color='black', linestyles='dashed')
                    if score_labels:
                        ax.text(abil + (x_max - x_min) / 100, y_max / 50, str(round(abil, 2)))
                        ax.text(x_min + (x_max - x_min) / 100, csem + y_max / 50, str(round(csem, 3)))

            if cat_highlight is not None and isinstance(items, str):
                item = items
                if cat_highlight in range(self.max_score_vector[item] + 1):
                    if cat_highlight == 0:
                        ax.axvspan(-100, self.thresholds_uncentred[item].iloc[0],
                                   facecolor='blue', alpha=0.2)
                    elif cat_highlight == self.max_score_vector[item]:
                        ax.axvspan(self.thresholds_uncentred[item].iloc[-1], 100,
                                   facecolor='blue', alpha=0.2)
                    else:
                        lo = self.thresholds_uncentred[item].iloc[cat_highlight - 1]
                        hi = self.thresholds_uncentred[item].iloc[cat_highlight]
                        if hi > lo:
                            ax.axvspan(lo, hi, facecolor='blue', alpha=0.2)

            if y_max <= 0:
                y_max = float(y_data.max()) * 1.1

            ax.set_xlim(x_min, x_max)
            ax.set_ylim(0, y_max)
            ax.set_xlabel('Ability', fontsize=axis_font_size, fontweight='bold', wrap=True)
            ax.set_ylabel(y_label, fontsize=axis_font_size, fontweight='bold', wrap=True)
            ax.set_title(graph_title, fontsize=title_font_size, fontweight='bold', wrap=True)
            ax.grid(True)
            ax.tick_params(axis='x', labelsize=labelsize)
            ax.tick_params(axis='y', labelsize=labelsize)

            if filename is not None:
                graph.savefig(filename + f'.{file_format}', dpi=plot_density)

            # Close the figure before returning. In Jupyter, two display events
            # fire if the figure is still open when returned: one from show()
            # and one from the notebook's auto-display of the returned object.
            # Closing here prevents the second. The figure data is preserved in
            # the returned object so callers can still inspect axes, save etc.
            plt.close(graph)

        return graph

    def icc(self, item, obs=False, no_of_classes=5, title=None,
            thresh_lines=False, central_diff=False, score_lines=None,
            score_labels=False, cat_highlight=None, xmin=-5, xmax=5,
            plot_style='white', palette='dark blue', black=False,
            font='Times New Roman', title_font_size=15, axis_font_size=12,
            labelsize=12, filename=None, file_format='png', dpi=300):
        """
        Plot the Item Characteristic Curve (ICC) for a single item.

        Displays modelled expected score as a function of person ability.
        Optionally overlays observed class-interval mean scores.

        Parameters
        ----------
        item : str
            Item identifier.
        obs : bool, default False
            If True, overlays observed class-interval mean scores.
        no_of_classes : int, default 5
            Number of class intervals for the observed overlay.
        title : str or None, default None
            Plot title.
        thresh_lines : bool, default False
            Draw vertical lines at each threshold.
        central_diff : bool, default False
            Draw a line at the central item difficulty.
        score_lines : list or None, default None
            Raw scores at which to draw reference lines.
        score_labels : bool, default False
            Annotate score line intersections.
        cat_highlight : int or None, default None
            Category to shade.
        xmin, xmax : float
            Ability axis limits.
        plot_style : str, default 'white'
            Background style.
        palette : str, default 'dark blue'
            Colour palette.
        black : bool, default False
            If True, renders in black.
        font : str, default 'Times New Roman'
            Font family.
        title_font_size, axis_font_size, labelsize : int
            Font sizes.
        filename : str or None, default None
            If provided, saves the plot.
        file_format : str, default 'png'
            Output format.
        dpi : int, default 300
            Output resolution.

        Returns
        -------
        matplotlib.figure.Figure
        """
        # BUG FIX: 'person_abiliites' -> 'person_abilities' (typo in original)
        if obs and not hasattr(self, 'person_abilities'):
            self.person_abils(warm_corr=False)

        xobsdata = yobsdata = np.array(np.nan)
        if obs:
            mean_abilities, obs_means = self.class_intervals(
                items=item, abilities=self.person_abilities, no_of_classes=no_of_classes
            )
            xobsdata = pd.Series(mean_abilities)
            yobsdata = np.array(obs_means).reshape(-1, 1)

        abilities = np.arange(-20, 20, 0.1)
        y = np.array([self.exp_score_uncentred(a, self.thresholds_uncentred[item])
                      for a in abilities]).reshape(-1, 1)

        return self.plot_data(
            x_data=abilities, y_data=y, x_obs_data=xobsdata, y_obs_data=yobsdata,
            x_min=xmin, x_max=xmax, y_max=self.max_score_vector[item], items=item,
            graph_title=title or '', y_label='Expected score', obs=obs,
            thresh_lines=thresh_lines, central_diff=central_diff,
            score_lines_item=[item, score_lines], score_labels=score_labels,
            plot_style=plot_style, palette=palette, cat_highlight=cat_highlight,
            black=black, font=font, title_font_size=title_font_size,
            axis_font_size=axis_font_size, labelsize=labelsize,
            filename=filename, plot_density=dpi, file_format=file_format
        )

    def crcs(self, item, obs=None, no_of_classes=5, title=None,
             thresh_lines=False, central_diff=False, cat_highlight=None,
             xmin=-5, xmax=5, plot_style='white', palette='dark blue', black=False,
             font='Times New Roman', title_font_size=15, axis_font_size=12,
             labelsize=12, filename=None, file_format='png', dpi=300):
        """
        Plot Category Response Curves (CRCs) for a single item.

        Displays the probability of each response category as a function of
        ability using uncentred PCM parameterisation. Optionally overlays
        observed class-interval category proportions.

        Parameters
        ----------
        item : str
            Item identifier.
        obs : list, 'all', or None, default None
            Observed overlay: 'all' for all categories, list of indices,
            or None for no overlay.
        no_of_classes : int, default 5
            Number of class intervals.
        title : str or None, default None
            Plot title.
        thresh_lines : bool, default False
            Draw vertical lines at each threshold.
        central_diff : bool, default False
            Draw a line at the central difficulty.
        cat_highlight : int or None, default None
            Category to shade.
        xmin, xmax : float
            Ability axis limits.
        plot_style, palette, black, font : see plot_data().
        title_font_size, axis_font_size, labelsize : int
            Font sizes.
        filename : str or None, default None
            If provided, saves the plot.
        file_format : str, default 'png'
            Output format.
        dpi : int, default 300
            Output resolution.

        Returns
        -------
        matplotlib.figure.Figure
        """
        # BUG FIX: typo 'person_abiliites' -> 'person_abilities'
        if obs is not None and not hasattr(self, 'person_abilities'):
            self.person_abils(warm_corr=False)
        if item == 'none':
            item = None

        xobsdata = yobsdata = np.array(np.nan)
        if obs is not None:
            mean_abilities, obs_props = self.class_intervals_cats(
                item=item, no_of_classes=no_of_classes
            )
            xobsdata, yobsdata = mean_abilities, obs_props
            if obs != 'all':
                if not all(c in np.arange(self.max_score_vector[item] + 1) for c in obs):
                    print("Invalid 'obs'. Valid values are None, 'all', or list of categories.")
                    return
                yobsdata = yobsdata[:, obs]

        abilities = np.arange(-20, 20, 0.1)
        y = np.array([[self.cat_prob_uncentred(a, cat, self.thresholds_uncentred[item])
                       for cat in range(self.max_score_vector[item] + 1)]
                      for a in abilities])

        return self.plot_data(
            x_data=abilities, y_data=y, x_min=xmin, x_max=xmax, y_max=1,
            x_obs_data=xobsdata, y_obs_data=yobsdata, items=item,
            graph_title=title or '', y_label='Probability', obs=obs,
            thresh_lines=thresh_lines, central_diff=central_diff,
            cat_highlight=cat_highlight, plot_style=plot_style, palette=palette,
            black=black, font=font, title_font_size=title_font_size,
            axis_font_size=axis_font_size, labelsize=labelsize,
            filename=filename, plot_density=dpi, file_format=file_format
        )

    def threshold_ccs(self, item, obs=None, no_of_classes=5, title=None,
                      thresh_lines=False, central_diff=False, cat_highlight=None,
                      xmin=-5, xmax=5, plot_style='white', palette='dark blue',
                      black=False, font='Times New Roman', title_font_size=15,
                      axis_font_size=12, labelsize=12, filename=None,
                      file_format='png', dpi=300):
        """
        Plot Threshold Characteristic Curves (TCCs) for a single item.

        Displays the probability of scoring in the higher of two adjacent
        categories at each threshold, as a function of person ability.

        Parameters
        ----------
        item : str
            Item identifier.
        obs : list, 'all', or None, default None
            Observed overlay: 'all' for all thresholds, list of 1-based
            threshold indices, or None.
        no_of_classes : int, default 5
            Number of class intervals.
        title : str or None, default None
            Plot title.
        thresh_lines : bool, default False
            Draw vertical lines at threshold locations.
        central_diff : bool, default False
            Draw a line at the central difficulty.
        cat_highlight : int or None, default None
            Threshold category to shade.
        xmin, xmax : float
            Ability axis limits.
        plot_style, palette, black, font : see plot_data().
        title_font_size, axis_font_size, labelsize : int
            Font sizes.
        filename : str or None, default None
            If provided, saves the plot.
        file_format : str, default 'png'
            Output format.
        dpi : int, default 300
            Output resolution.

        Returns
        -------
        matplotlib.figure.Figure
        """
        if obs is not None and not hasattr(self, 'person_abilities'):
            self.person_abils(warm_corr=False)

        xobsdata = yobsdata = np.array(np.nan)
        if obs is not None:
            mean_abilities, obs_props = self.class_intervals_thresholds(
                item, no_of_classes=no_of_classes
            )
            xobsdata, yobsdata = mean_abilities, obs_props
            if obs != 'all':
                if not all(c in np.arange(self.max_score_vector[item]) + 1 for c in obs):
                    print("Invalid 'obs'. Valid values are None, 'all', or list of categories.")
                    return
                obs_idx = [o - 1 for o in obs]
                xobsdata = xobsdata[:, obs_idx]
                yobsdata = yobsdata[:, obs_idx]

        abilities = np.arange(-20, 20, 0.1)
        y = np.array([[1.0 / (1.0 + np.exp(thr - a))
                       for thr in self.thresholds_uncentred[item]]
                      for a in abilities])

        return self.plot_data(
            x_data=abilities, y_data=y, y_max=1, x_min=xmin, x_max=xmax, items=item,
            x_obs_data=xobsdata, y_obs_data=yobsdata, graph_title=title or '',
            y_label='Probability', obs=obs, thresh_lines=thresh_lines,
            central_diff=central_diff, cat_highlight=cat_highlight,
            plot_style=plot_style, palette=palette, black=black, font=font,
            title_font_size=title_font_size, axis_font_size=axis_font_size,
            labelsize=labelsize, filename=filename, plot_density=dpi,
            file_format=file_format
        )

    def iic(self, item, ymax=None, thresh_lines=False, central_diff=False,
            point_info_lines=None, point_info_labels=False, cat_highlight=None,
            title=None, xmin=-5, xmax=5, plot_style='white', palette='dark blue',
            black=False, font='Times New Roman', title_font_size=15, axis_font_size=12,
            labelsize=12, filename=None, file_format='png', dpi=300):
        """
        Plot the Item Information Curve (IIC) for a single item.

        Displays Fisher information (item variance) as a function of ability
        using uncentred threshold parameterisation.

        Parameters
        ----------
        item : str
            Item identifier.
        ymax : float or None, default None
            Upper y-axis limit. Auto-scaled if None.
        thresh_lines : bool, default False
            Draw vertical lines at each threshold.
        central_diff : bool, default False
            Draw a line at the central difficulty.
        point_info_lines : list or None, default None
            Ability values at which to draw information reference lines.
        point_info_labels : bool, default False
            Annotate information line intersections.
        cat_highlight : int or None, default None
            Category to shade.
        title : str or None, default None
            Plot title.
        xmin, xmax : float
            Ability axis limits.
        plot_style, palette, black, font : see plot_data().
        title_font_size, axis_font_size, labelsize : int
            Font sizes.
        filename : str or None, default None
            If provided, saves the plot.
        file_format : str, default 'png'
            Output format.
        dpi : int, default 300
            Output resolution.

        Returns
        -------
        matplotlib.figure.Figure
        """
        abilities = np.arange(-20, 20, 0.1)
        y = np.array([self.variance_uncentred(a, self.thresholds_uncentred[item])
                      for a in abilities]).reshape(-1, 1)
        if ymax is None:
            ymax = float(y.max()) * 1.1

        return self.plot_data(
            x_data=abilities, y_data=y, x_min=xmin, x_max=xmax, y_max=ymax,
            thresh_lines=thresh_lines, items=item, central_diff=central_diff,
            point_info_lines_item=[item, point_info_lines],
            score_labels=point_info_labels, cat_highlight=cat_highlight,
            graph_title=title or '', y_label='Fisher information',
            plot_style=plot_style, palette=palette, black=black, font=font,
            title_font_size=title_font_size, axis_font_size=axis_font_size,
            labelsize=labelsize, filename=filename, file_format=file_format,
            plot_density=dpi
        )

    def tcc(self, items=None, obs=False, xmin=-5, xmax=5, no_of_classes=5,
            title=None, score_lines=None, score_labels=False, warm=True,
            plot_style='white', palette='dark blue', black=False,
            font='Times New Roman', title_font_size=15, axis_font_size=12,
            labelsize=12, filename=None, file_format='png', dpi=300):
        """
        Plot the Test Characteristic Curve (TCC).

        Displays expected total score as a function of ability. Optionally
        overlays observed class-interval mean total scores.

        Parameters
        ----------
        items : str, list, or None, default None
            Item subset. None uses all items.
        obs : bool, default False
            If True, overlays observed mean total scores.
        xmin, xmax : float
            Ability axis limits.
        no_of_classes : int, default 5
            Number of class intervals for observed overlay.
        title : str or None, default None
            Plot title.
        score_lines : list or None, default None
            Raw total scores at which to draw reference lines.
        score_labels : bool, default False
            Annotate score line intersections.
        warm : bool, default True
            Passed for API consistency.
        plot_style, palette, black, font : see plot_data().
        title_font_size, axis_font_size, labelsize : int
            Font sizes.
        filename : str or None, default None
            If provided, saves the plot.
        file_format : str, default 'png'
            Output format.
        dpi : int, default 300
            Output resolution.

        Returns
        -------
        matplotlib.figure.Figure
        """
        if isinstance(items, str) and items in ('all', 'none'):
            items = None
        elif isinstance(items, str):
            items = [items]

        # BUG FIX: typo 'person_abiliites' -> 'person_abilities'
        if obs and not hasattr(self, 'person_abilities'):
            self.person_abils(warm_corr=False)

        xobsdata = yobsdata = np.array(np.nan)
        if obs:
            mean_abilities, obs_means = self.class_intervals(
                items=items, abilities=self.person_abilities, no_of_classes=no_of_classes
            )
            xobsdata = mean_abilities
            yobsdata = np.array(obs_means).reshape(no_of_classes, 1)

        abilities = np.arange(-20, 20, 0.1)
        item_keys = self.dataframe.columns if items is None else items
        y = np.array([sum(self.exp_score_uncentred(a, self.thresholds_uncentred[it])
                          for it in item_keys)
                      for a in abilities]).reshape(-1, 1)
        y_max = sum(self.max_score_vector[it] for it in item_keys)

        return self.plot_data(
            x_data=abilities, y_data=y, x_obs_data=xobsdata, y_obs_data=yobsdata,
            x_min=xmin, x_max=xmax, y_max=y_max, items=items,
            score_lines_test=score_lines, score_labels=score_labels, warm=warm,
            graph_title=title or '', y_label='Expected score', obs=obs,
            plot_style=plot_style, palette=palette, black=black, font=font,
            title_font_size=title_font_size, axis_font_size=axis_font_size,
            labelsize=labelsize, filename=filename, file_format=file_format,
            plot_density=dpi
        )

    def test_info(self, items=None, point_info_lines=None, point_info_labels=False,
                  xmin=-5, xmax=5, ymax=None, title=None, plot_style='white',
                  palette='dark blue', black=False, font='Times New Roman',
                  title_font_size=15, axis_font_size=12, labelsize=12,
                  filename=None, file_format='png', dpi=300):
        """
        Plot the Test Information Curve.

        Displays sum of item Fisher information values as a function of ability.

        Parameters
        ----------
        items : str, list, or None, default None
            Item subset. None uses all items.
        point_info_lines : list or None, default None
            Ability values at which to draw reference lines.
        point_info_labels : bool, default False
            Annotate information line intersections.
        xmin, xmax : float
            Ability axis limits.
        ymax : float or None, default None
            Upper y-axis limit. Auto-scaled if None.
        title : str or None, default None
            Plot title.
        plot_style, palette, black, font : see plot_data().
        title_font_size, axis_font_size, labelsize : int
            Font sizes.
        filename : str or None, default None
            If provided, saves the plot.
        file_format : str, default 'png'
            Output format.
        dpi : int, default 300
            Output resolution.

        Returns
        -------
        matplotlib.figure.Figure
        """
        if isinstance(items, str) and items in ('all', 'none'):
            items = None
        elif isinstance(items, str):
            items = [items]
        item_keys = self.dataframe.columns if items is None else items
        abilities = np.arange(-20, 20, 0.1)
        y = np.array([sum(self.variance_uncentred(a, self.thresholds_uncentred[it])
                          for it in item_keys)
                      for a in abilities]).reshape(-1, 1)
        if ymax is None:
            ymax = float(y.max()) * 1.1

        return self.plot_data(
            x_data=abilities, y_data=y, x_min=xmin, x_max=xmax, y_max=ymax,
            items=items, graph_title=title or '',
            point_info_lines_test=point_info_lines, score_labels=point_info_labels,
            y_label='Fisher information', plot_style=plot_style, palette=palette,
            black=black, font=font, title_font_size=title_font_size,
            axis_font_size=axis_font_size, labelsize=labelsize,
            filename=filename, file_format=file_format, plot_density=dpi
        )

    def test_csem(self, items=None, point_csem_lines=None, point_csem_labels=False,
                  xmin=-5, xmax=5, ymax=5, title=None, plot_style='white',
                  palette='dark blue', black=False, font='Times New Roman',
                  title_font_size=15, axis_font_size=12, labelsize=12,
                  filename=None, file_format='png', dpi=300):
        """
        Plot the Test Conditional Standard Error of Measurement (CSEM) Curve.

        Displays 1 / sqrt(I(theta)) as a function of ability, where I(theta)
        is total test information.

        Parameters
        ----------
        items : str, list, or None, default None
            Item subset. None uses all items.
        point_csem_lines : list or None, default None
            Ability values at which to draw CSEM reference lines.
        point_csem_labels : bool, default False
            Annotate CSEM line intersections.
        xmin, xmax : float
            Ability axis limits.
        ymax : float, default 5
            Upper y-axis limit.
        title : str or None, default None
            Plot title.
        plot_style, palette, black, font : see plot_data().
        title_font_size, axis_font_size, labelsize : int
            Font sizes.
        filename : str or None, default None
            If provided, saves the plot.
        file_format : str, default 'png'
            Output format.
        dpi : int, default 300
            Output resolution.

        Returns
        -------
        matplotlib.figure.Figure
        """
        if isinstance(items, str) and items in ('all', 'none'):
            items = None
        elif isinstance(items, str):
            items = [items]
        item_keys = self.dataframe.columns if items is None else items
        abilities = np.arange(-20, 20, 0.1)
        info = np.array([sum(self.variance_uncentred(a, self.thresholds_uncentred[it])
                             for it in item_keys)
                         for a in abilities])
        y = (1.0 / (info ** 0.5)).reshape(-1, 1)

        return self.plot_data(
            x_data=abilities, y_data=y, x_min=xmin, x_max=xmax, y_max=ymax,
            items=items, graph_title=title or '', point_csem_lines=point_csem_lines,
            score_labels=point_csem_labels, y_label='Conditional SEM',
            plot_style=plot_style, palette=palette, black=black, font=font,
            title_font_size=title_font_size, axis_font_size=axis_font_size,
            labelsize=labelsize, filename=filename, file_format=file_format,
            plot_density=dpi
        )

    def std_residuals_plot(self,
                           items=None,
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
        """
        Plot a histogram of standardised residuals.

        Displays the distribution of standardised residuals across all
        person-item combinations (or a subset of items). Under a well-fitting
        Rasch model these should approximate a standard normal distribution.
        Optionally overlays a standard normal density curve.

        Requires fit_statistics() to have been run first.

        Parameters
        ----------
        items : str, list, or None, default None
            Item subset. None uses all items.
        bin_width : float, default 0.5
            Width of histogram bins.
        x_min : float, default -6
            Left x-axis limit.
        x_max : float, default 6
            Right x-axis limit.
        normal : bool, default False
            If True, overlays a standard normal density curve.
        title : str or None, default None
            Plot title.
        plot_style : str, default 'white'
            Background style.
        font : str, default 'Times New Roman'
            Font family.
        title_font_size, axis_font_size, labelsize : int
            Font sizes.
        filename : str or None, default None
            If provided, saves the plot.
        file_format : str, default 'png'
            Output format.
        plot_density : int, default 300
            Output resolution.

        Returns
        -------
        matplotlib.figure.Figure
        """
        if isinstance(items, str) and items in ('all', 'none'):
            items = None
        elif isinstance(items, str):
            items = [items]

        std_residual_df = (self.std_residual_df if items is None
                           else self.std_residual_df[items])
        std_residual_list = std_residual_df.unstack().dropna()

        return self.std_residuals_hist(
            std_residual_list, bin_width=bin_width, x_min=x_min, x_max=x_max,
            normal=normal, title=title, plot_style=plot_style, font=font,
            title_font_size=title_font_size, axis_font_size=axis_font_size,
            labelsize=labelsize, filename=filename, file_format=file_format,
            plot_density=plot_density
        )