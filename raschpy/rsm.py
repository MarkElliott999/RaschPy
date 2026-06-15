from math import log
import warnings

import numpy as np
import pandas as pd
from scipy.stats import hmean, norm
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from matplotlib import colors as colors
from matplotlib import cm as cmx
import seaborn as sns

from raschpy.base import Rasch


class RSM(Rasch):

    def __init__(self,
                 dataframe,
                 max_score=None,
                 extreme_persons=True,
                 no_of_classes=5):

        self.max_score = int(np.nanmax(dataframe)) if max_score is None else max_score

        if extreme_persons:
            self.invalid_responses = dataframe[dataframe.isna().all(axis=1)]
            self.dataframe = dataframe[~dataframe.isna().all(axis=1)]
        else:
            scores = dataframe.sum(axis=1)
            # notna() mask is cleaner than (df == df) for detecting valid cells
            max_scores = dataframe.notna().sum(axis=1) * self.max_score
            self.invalid_responses = dataframe[(scores == 0) | (scores == max_scores)]
            self.dataframe = dataframe[(scores > 0) & (scores < max_scores)]

        self.no_of_items = self.dataframe.shape[1]
        self.items = self.dataframe.columns
        self.no_of_persons = self.dataframe.shape[0]
        self.persons = self.dataframe.index
        self.no_of_classes = no_of_classes

    """
    Rating Scale Model (Andrich 1978) formulation of the polytomous Rasch model.

    The RSM constrains all items to share the same set of Rasch-Andrich
    threshold parameters (tau_1..tau_m), differing only in their central
    difficulties (delta_i). Thresholds are estimated using CPAT
    (Elliott & Buttery, 2022); item difficulties are estimated using PAIR.

    Threshold convention: self.thresholds is a numpy array of length max_score+1,
    where thresholds[0] = 0 (tau_0 sentinel) and thresholds[1..max_score] are
    the Rasch-Andrich threshold parameters tau_1..tau_m.
    """

    # ------------------------------------------------------------------
    # Core probability / expected-score functions (scalar, used in plots)
    # ------------------------------------------------------------------

    def cat_prob(self, ability, difficulty, category, thresholds):
        """
        Compute the probability of a response category (centred RSM parameterisation).

        Log-numerator for category k: k*(b-d) - cumsum(tau)[k], where b is ability,
        d is item difficulty, tau[0]=0 sentinel. Numerically stabilised via log-sum-exp.

        Parameters
        ----------
        ability : float
            Person ability on the logit scale.
        difficulty : float
            Item difficulty (central location) on the logit scale.
        category : int
            Response category (0 to max_score).
        thresholds : array-like
            Rasch-Andrich threshold vector, length max_score+1, thresholds[0]=0.

        Returns
        -------
        float
            Probability of the specified category, in [0, 1].
        """
        cats = np.arange(len(thresholds), dtype=float)
        cumsum = np.cumsum(thresholds)
        log_nums = cats * (ability - difficulty) - cumsum
        log_nums -= log_nums.max()
        nums = np.exp(log_nums)
        return nums[category] / nums.sum()

    def exp_score(self, ability, difficulty, thresholds):
        """
        Compute the expected score on an item.

        Calculates E[X | ability, difficulty, thresholds] = sum(k * P(X=k))
        using the RSM centred parameterisation. Numerically stabilised.

        Parameters
        ----------
        ability : float
            Person ability on the logit scale.
        difficulty : float
            Item difficulty on the logit scale.
        thresholds : array-like
            Rasch-Andrich threshold vector, length max_score+1.

        Returns
        -------
        float
            Expected score in [0, max_score].
        """
        cats = np.arange(len(thresholds), dtype=float)
        cumsum = np.cumsum(thresholds)
        log_nums = cats * (ability - difficulty) - cumsum
        log_nums -= log_nums.max()
        nums = np.exp(log_nums)
        probs = nums / nums.sum()
        return (cats * probs).sum()

    def variance(self, ability, difficulty, thresholds):
        """
        Compute item variance (Fisher information).

        Calculates Var[X | ability, difficulty, thresholds] = sum((k - E[X])^2 * P(X=k)).
        Equal to the Fisher information at the given ability.

        Parameters
        ----------
        ability : float
            Person ability on the logit scale.
        difficulty : float
            Item difficulty on the logit scale.
        thresholds : array-like
            Rasch-Andrich threshold vector, length max_score+1.

        Returns
        -------
        float
            Item variance / Fisher information. Always non-negative.
        """
        cats = np.arange(len(thresholds), dtype=float)
        cumsum = np.cumsum(thresholds)
        log_nums = cats * (ability - difficulty) - cumsum
        log_nums -= log_nums.max()
        nums = np.exp(log_nums)
        probs = nums / nums.sum()
        expected = (cats * probs).sum()
        return ((cats - expected) ** 2 * probs).sum()

    def kurtosis(self, ability, difficulty, thresholds):
        """
        Compute the fourth central moment of the response distribution.

        Calculates sum((k - E[X])^4 * P(X=k)) using the RSM centred
        parameterisation. Used in the Wilson-Hilferty approximation for
        standardised fit statistics (Infit Z, Outfit Z).

        Parameters
        ----------
        ability : float
            Person ability on the logit scale.
        difficulty : float
            Item difficulty on the logit scale.
        thresholds : array-like
            Rasch-Andrich threshold vector, length max_score+1.

        Returns
        -------
        float
            Fourth central moment of the response distribution.
        """
        cats = np.arange(len(thresholds), dtype=float)
        cumsum = np.cumsum(thresholds)
        log_nums = cats * (ability - difficulty) - cumsum
        log_nums -= log_nums.max()
        nums = np.exp(log_nums)
        probs = nums / nums.sum()
        expected = (cats * probs).sum()
        return ((cats - expected) ** 4 * probs).sum()

    # ------------------------------------------------------------------
    # Vectorised category probability engine
    # ------------------------------------------------------------------

    def _cat_probs_matrix(self, abilities, difficulties, thresholds):
        '''
        Vectorised RSM category probability computation.

        The RSM log-numerator for category k, person n, item i is:
            k * (ability_n - difficulty_i) - cumsum(thresholds)[k]
        where cumsum(thresholds)[k] = sum(thresholds[0..k]).

        Because thresholds are SHARED across items (unlike PCM), cumsum is
        identical for all items and the full (K+1, N, I) tensor is computed
        in a single broadcast without any Python loop over items or categories.

        Returns
        -------
        probs    : ndarray (K+1, N, I)  -- category probabilities
        cats_arr : ndarray (K+1,)       -- category indices [0..max_score]
        '''
        cats_arr = np.arange(len(thresholds), dtype=float)  # (K+1,)
        cumsum = np.cumsum(thresholds)                       # (K+1,)
        ab   = np.asarray(abilities,    dtype=float)         # (N,)
        diff = np.asarray(difficulties, dtype=float)         # (I,)

        # log_num[k, n, i] = k*(ab[n] - diff[i]) - cumsum[k]
        log_num = (cats_arr[:, None, None]
                   * (ab[None, :, None] - diff[None, None, :])
                   - cumsum[:, None, None])                  # (K+1, N, I)

        # Numerically stable softmax along category axis
        log_num -= log_num.max(axis=0, keepdims=True)
        probs = np.exp(log_num)
        probs /= probs.sum(axis=0, keepdims=True)

        return probs, cats_arr

    # ------------------------------------------------------------------
    # CPAT threshold estimation
    # ------------------------------------------------------------------

    def _threshold_distance(self, threshold, difficulties, constant=0.1):
        """
        Estimate the distance between adjacent Rasch-Andrich thresholds (CPAT).

        Implements Elliott & Buttery (2022). For threshold k, counts:
          num[i,j]: persons scoring exactly k on both items i and j
          den[i,j]: persons scoring k-1 on item i and k+1 on item j
        Conditioning on these patterns removes person ability, leaving a
        contrast identifying the threshold location. Harmonic mean weighting
        downweights near-zero counts. Vectorised via matrix multiplication.

        Parameters
        ----------
        threshold : int
            1-based threshold index (1 to max_score-1).
        difficulties : pandas.Series
            Item difficulty estimates indexed by item name.
        constant : float, default 0.1
            Additive smoothing constant for zero cells.

        Returns
        -------
        float
            Estimated distance between tau_threshold and tau_{threshold+1}, in logits.
            Returns numpy.nan if total weight is zero.
        """
        df_array = np.array(self.dataframe, dtype=np.float64)

        # Build (N, I) indicator arrays for each relevant score value
        at_k   = (df_array == threshold).astype(np.float64)      # X == k
        at_km1 = (df_array == threshold - 1).astype(np.float64)  # X == k-1
        at_kp1 = (df_array == threshold + 1).astype(np.float64)  # X == k+1

        # (I, I) count matrices via matrix multiplication
        num_matrix = at_k.T @ at_k        # count(X_i==k AND X_j==k)
        den_matrix = at_km1.T @ at_kp1   # count(X_i==k-1 AND X_j==k+1)

        valid = (num_matrix + den_matrix) > 0
        num_s = np.where(valid, num_matrix + constant, 0.0)
        den_s = np.where(valid, den_matrix + constant, 0.0)

        # Harmonic mean weight: 2*a*b/(a+b), zero where invalid
        with np.errstate(divide='ignore', invalid='ignore'):
            weight_matrix = np.where(valid,
                                     2.0 * num_s * den_s / (num_s + den_s),
                                     0.0)

        # Difficulty contrast matrix: delta_i - delta_j  shape (I, I)
        diffs = difficulties.values
        diff_matrix = diffs[:, None] - diffs[None, :]

        # Log frequency ratio + difficulty contrast, weighted
        with np.errstate(divide='ignore', invalid='ignore'):
            log_ratio = np.where(valid, np.log(num_s) - np.log(den_s), 0.0)

        total_weight = weight_matrix.sum()
        if total_weight == 0:
            return np.nan

        return (weight_matrix * (log_ratio + diff_matrix)).sum() / total_weight

    def threshold_set(self, difficulties, constant=0.1):
        """
        Compute the full Rasch-Andrich threshold vector from CPAT distances.

        Chains the m-1 CPAT distances via cumulative sum, mean-centres the
        result, and prepends tau_0=0 as a sentinel. Called by calibrate().

        Parameters
        ----------
        difficulties : pandas.Series
            Item difficulties from Stage 1 (PAIR), indexed by item name.
        constant : float, default 0.1
            Smoothing constant passed to _threshold_distance().

        Returns
        -------
        numpy.ndarray
            Threshold vector of length max_score+1, thresholds[0]=0,
            thresholds[1..max_score] centred at 0.
        """
        thresh_distances = [
            self._threshold_distance(threshold + 1, difficulties, constant)
            for threshold in range(self.max_score - 1)
        ]

        # Chain distances: thresh[k] = sum of distances[0..k-1]
        thresholds = np.array([sum(thresh_distances[:t])
                                for t in range(self.max_score)])
        thresholds -= np.mean(thresholds)
        # Insert tau_0 = 0 sentinel at position 0
        return np.insert(thresholds, 0, 0.0)

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate(self,
                  constant=0.1,
                  method='cos',
                  matrix_power=3,
                  log_lik_tol=0.000001):
        """
        Two-stage RSM calibration: PAIR item difficulties + CPAT thresholds.

        Stage 1: Builds a pairwise contingency matrix (entry (i,j) = count of
        persons scoring one point higher on item i than j), resolves structural
        zeroes via matrix powers, and extracts item difficulties with
        priority_vector().

        Stage 2: Given item difficulties, estimates adjacent-threshold distances
        via CPAT (_threshold_distance()), chains them, and centres the result.

        Issues a UserWarning if only one item is present or if constant=0 and
        any item has all-maximum scores.

        Parameters
        ----------
        constant : float, default 0.1
            Additive smoothing constant for zero cells.
        method : str, default 'cos'
            Priority vector extraction method.
        matrix_power : int, default 3
            Initial matrix power for PAIR zero-resolution.
        log_lik_tol : float, default 0.000001
            Log-likelihood convergence tolerance for priority vector extraction.

        Attributes set
        --------------
        diffs : pandas.Series
            Item difficulty estimates, zero-centred.
        thresholds : numpy.ndarray
            Shared Rasch-Andrich threshold vector, length max_score+1,
            thresholds[0]=0, thresholds[1..max_score] centred at 0.
        null_persons : pandas.Index
            Persons dropped (entirely missing data).
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

        self.null_persons = self.dataframe.index[self.dataframe.isnull().all(1)]
        self.dataframe = self.dataframe.drop(self.null_persons)
        self.no_of_persons = self.dataframe.shape[0]

        df_array = self.dataframe.to_numpy(dtype=np.float64)

        # Stage 1: PAIR item difficulty matrix.
        # matrix[i,j] = count(X_i == X_j + 1) across all persons.
        # Vectorised: for each adjacent score pair (s, s-1), build indicator
        # matrices and accumulate -- but for simplicity the original loop
        # structure is fast enough since it's O(I^2) comparisons at C speed.
        matrix = np.array([
            [np.count_nonzero(df_array[:, i] == df_array[:, j] + 1)
             for j in range(self.no_of_items)]
            for i in range(self.no_of_items)
        ], dtype=np.float64)

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

        self.diffs = self.priority_vector(mat, method=method,
                                          log_lik_tol=log_lik_tol)

        # Stage 2: CPAT threshold estimation
        self.thresholds = self.threshold_set(self.diffs, constant=constant)

    # ------------------------------------------------------------------
    # Standard errors (bootstrap)
    # ------------------------------------------------------------------

    def std_errors(self,
                   interval=None,
                   no_of_samples=100,
                   constant=0.1,
                   method='cos',
                   matrix_power=3,
                   log_lik_tol=0.000001):
        """
        Estimate bootstrap standard errors for item difficulties and thresholds.

        Draws no_of_samples bootstrap resamples of person-level response data,
        calibrates each, and computes SDs of item difficulty and threshold
        estimates across samples. Also computes category width SEs.

        Parameters
        ----------
        interval : float or None, default None
            CI width (e.g. 0.95). If None, only SEs computed.
        no_of_samples : int, default 100
            Number of bootstrap resamples.
        constant : float, default 0.1
            Smoothing constant for bootstrap calibrations.
        method : str, default 'cos'
            Priority vector extraction method.
        matrix_power : int, default 3
            Matrix power for bootstrap calibrations.
        log_lik_tol : float, default 0.000001
            Convergence tolerance.

        Attributes set
        --------------
        item_se : pandas.Series
            Bootstrap SE for each item difficulty.
        threshold_se : numpy.ndarray
            Bootstrap SE for each threshold (length max_score+1; index 0 = 0).
        cat_width_se : pandas.Series
            Bootstrap SE for each category width (threshold spacing).
        item_low / item_high : pandas.Series or None
            Bootstrap CI bounds for item difficulties.
        threshold_low / threshold_high : numpy.ndarray or None
            Bootstrap CI bounds for thresholds.
        item_bootstrap : pandas.DataFrame
            Bootstrap item difficulty estimates, shape (no_of_samples, items).
        threshold_bootstrap : pandas.DataFrame
            Bootstrap threshold estimates, shape (no_of_samples, max_score+1).
        cat_width_bootstrap : pandas.DataFrame
            Bootstrap category width estimates.
        """
        samples = [RSM(self.dataframe.sample(frac=1, replace=True),
                       max_score=self.max_score)
                   for _ in range(no_of_samples)]

        for sample in samples:
            sample.calibrate(constant=constant, method=method,
                             matrix_power=matrix_power,
                             log_lik_tol=log_lik_tol)

        item_ests      = np.array([s.diffs.values     for s in samples])  # (B, I)
        threshold_ests = np.array([s.thresholds        for s in samples])  # (B, K+1)

        sample_idx = [f'Sample {i + 1}' for i in range(no_of_samples)]

        self.item_bootstrap = pd.DataFrame(item_ests,
                                           index=sample_idx,
                                           columns=self.dataframe.columns)
        self.item_se  = pd.Series(np.nanstd(item_ests, axis=0),
                                  index=self.dataframe.columns)

        self.threshold_bootstrap = pd.DataFrame(
            threshold_ests, index=sample_idx,
            columns=range(self.max_score + 1)
        )
        self.threshold_se = np.nanstd(threshold_ests, axis=0)

        # Category width bootstrap: width_k = tau_{k+1} - tau_k
        cat_widths = {cat + 1: threshold_ests[:, cat + 2] - threshold_ests[:, cat + 1]
                      for cat in range(self.max_score - 1)}
        self.cat_width_bootstrap = pd.DataFrame(cat_widths, index=sample_idx)
        self.cat_width_bootstrap.columns = range(1, self.max_score)
        self.cat_width_se = {cat: np.nanstd(est)
                             for cat, est in cat_widths.items()}
        self.cat_width_se = pd.Series(self.cat_width_se)

        if interval is not None:
            lo, hi = 50 * (1 - interval), 50 * (1 + interval)
            self.item_low  = pd.Series(np.percentile(item_ests, lo, axis=0),
                                       index=self.dataframe.columns)
            self.item_high = pd.Series(np.percentile(item_ests, hi, axis=0),
                                       index=self.dataframe.columns)
            self.threshold_low  = np.percentile(threshold_ests, lo, axis=0)
            self.threshold_high = np.percentile(threshold_ests, hi, axis=0)
            self.cat_width_low  = {cat: np.percentile(est, lo)
                                   for cat, est in cat_widths.items()}
            self.cat_width_high = {cat: np.percentile(est, hi)
                                   for cat, est in cat_widths.items()}
        else:
            self.item_low = self.item_high = None
            self.threshold_low = self.threshold_high = None
            self.cat_width_low = self.cat_width_high = None

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

        Iteratively solves sum_i(E[X_i | b]) = observed_score using the shared
        RSM threshold parameterisation and vectorised _cat_probs_matrix. Per-person
        convergence tracking prevents runaway estimates. Extreme scores are
        adjusted. Optionally applies Warm's (1989) bias correction.

        Parameters
        ----------
        persons : str or list
            Person identifier(s). Pass 'all' for all persons.
        items : str, list, or None, default None
            Item subset. None uses all items.
        warm_corr : bool, default True
            Warm bias correction.
        tolerance : float, default 0.00001
            Convergence criterion per person.
        max_iters : int, default 100
            Maximum iterations. Non-converged persons set to NaN.
        ext_score_adjustment : float, default 0.5
            Adjustment for extreme scores.

        Returns
        -------
        pandas.Series
            Ability estimates indexed by person identifier, in logits.
        """
        if isinstance(persons, str):
            persons = self.persons if persons == 'all' else [persons]

        if items is None:
            items = list(self.items)
        elif isinstance(items, str):
            items = list(self.items) if items == 'all' else [items]

        difficulties = self.diffs.loc[items]
        person_data  = self.dataframe.loc[persons, items]
        person_filter = person_data.notna().astype(float)

        scores     = person_data.sum(axis=1).astype(float)
        ext_scores = person_filter.sum(axis=1) * self.max_score

        # Adjust extreme scores to keep log() finite
        scores[scores == 0]          = ext_score_adjustment
        scores[scores == ext_scores] -= ext_score_adjustment

        mean_diff = difficulties.mean()

        try:
            estimates = np.log(scores) - np.log(ext_scores - scores) + mean_diff

            # Per-person convergence mask — freeze persons once change < tolerance.
            # Without this, the log-sum-exp implementation (which gives numerically
            # valid probs for all ability values) keeps updating slowly-converging
            # persons every iteration, allowing drift of ±1 logit per step.
            active = pd.Series(True, index=list(persons))
            iters  = 0

            diff_arr = difficulties.values  # (I,)

            while active.any() and iters <= max_iters:
                active_idx = active[active].index

                probs, cats_arr = self._cat_probs_matrix(
                    estimates.loc[active_idx].values, diff_arr, self.thresholds
                )
                # probs: (K+1, N_active, I)
                exp_score = (cats_arr[:, None, None] * probs).sum(axis=0)   # (N_active, I)
                exp_df = pd.DataFrame(exp_score, index=active_idx, columns=items)
                exp_df *= person_filter.loc[active_idx]

                dev  = cats_arr[:, None, None] - exp_score[None, :, :]      # (K+1, N_active, I)
                info = (dev ** 2 * probs).sum(axis=0)                        # (N_active, I)
                info_df = pd.DataFrame(info, index=active_idx, columns=items)
                info_df *= person_filter.loc[active_idx]

                result_list = exp_df.sum(axis=1)
                info_list   = info_df.sum(axis=1)

                changes = ((result_list - scores.loc[active_idx]) / info_list).clip(-1, 1)
                estimates.loc[active_idx] -= changes

                active.loc[active_idx] = abs(changes) > tolerance
                iters += 1

            if iters >= max_iters and active.any():
                n_nc = int(active.sum())
                warnings.warn(
                    f'{n_nc} person(s) did not converge in abil() and will be set to NaN. '
                    f'Consider increasing max_iters or checking for degenerate response patterns.',
                    UserWarning, stacklevel=2
                )
                estimates[active] = np.nan

            if warm_corr:
                valid = estimates.notna()
                if valid.any():
                    estimates[valid] += self.warm(
                        estimates[valid], items,
                        person_filter.loc[estimates.index[valid]]
                    )

        except Exception as e:
            warnings.warn(f'abil() failed with exception: {e}. '
                          'Returning NaN for all persons.',
                          UserWarning, stacklevel=2)
            estimates = pd.Series(np.nan, index=list(persons))

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
            Convergence tolerance.
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
            self.persons, items=items, warm_corr=warm_corr,
            tolerance=tolerance, max_iters=max_iters,
            ext_score_adjustment=ext_score_adjustment
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
        items : str, list, or None, default None
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
        if items is None or (isinstance(items, str) and items == 'all'):
            items = list(self.items)
        elif isinstance(items, str):
            items = [items]

        difficulties = self.diffs.loc[items]
        ext_score    = len(items) * self.max_score
        mean_diff    = difficulties.mean()

        used_score = float(score)
        if used_score == 0:
            used_score = ext_score_adjustment
        elif used_score == ext_score:
            used_score -= ext_score_adjustment

        estimate = log(used_score) - log(ext_score - used_score) + mean_diff
        change, iters = 1.0, 0

        while abs(change) > tolerance and iters <= max_iters:
            result = sum(self.exp_score(estimate, diff, self.thresholds)
                         for diff in difficulties)
            info   = sum(self.variance(estimate, diff, self.thresholds)
                         for diff in difficulties)
            change = max(-1.0, min(1.0, (result - used_score) / info))
            estimate -= change
            iters += 1

        if warm_corr:
            pf = pd.DataFrame(1.0, columns=items, index=[score])
            estimate += float(self.warm(pd.Series({score: estimate}), items, pf).iloc[0])

        if iters >= max_iters:
            warnings.warn(
                'Maximum iterations reached before convergence in score_abil(). '
                'Returned estimate may be inaccurate.',
                UserWarning, stacklevel=2
            )

        return estimate

    def abil_lookup_table(self,
                          items=None,
                          ext_scores=True,
                          warm_corr=True,
                          tolerance=0.00001,
                          max_iters=100,
                          ext_score_adjustment=0.5):
        """
        Build a score-to-ability lookup table for all possible raw scores.

        Parameters
        ----------
        items : str, list, or None, default None
            Item subset. None uses all items.
        ext_scores : bool, default True
            If True, includes extreme scores adjusted by ext_score_adjustment.
        warm_corr : bool, default True
            Warm bias correction.
        tolerance : float, default 0.00001
            Convergence tolerance.
        max_iters : int, default 100
            Maximum iterations.
        ext_score_adjustment : float, default 0.5
            Adjustment for extreme scores.

        Attributes set
        --------------
        abil_table : pandas.Series
            Ability estimate for each possible raw score, indexed by score.
        """
        if isinstance(items, str) and items in ('all', 'none'):
            items = None
        elif isinstance(items, str):
            items = [items]
        if items is None:
            items = list(self.items)

        no_of_items  = len(items)
        difficulties = self.diffs.loc[items]
        total_max    = no_of_items * self.max_score

        if ext_scores:
            scores      = np.arange(total_max + 1)
            used_scores = scores.astype(float)
            used_scores[0]  += ext_score_adjustment
            used_scores[-1] -= ext_score_adjustment
        else:
            scores      = np.arange(1, total_max)
            used_scores = scores.astype(float)

        mean_diff = difficulties.mean()
        estimates = pd.Series(
            np.log(used_scores) - np.log(total_max - used_scores) + mean_diff,
            index=scores
        )

        changes = pd.Series(1.0, index=scores)
        iters   = 0
        diff_arr = difficulties.values

        while abs(changes).max() > tolerance and iters <= max_iters:
            probs, cats_arr = self._cat_probs_matrix(
                estimates.values, diff_arr, self.thresholds
            )
            exp_score = (cats_arr[:, None, None] * probs).sum(axis=0)
            exp_df    = pd.DataFrame(exp_score, index=scores, columns=items)

            dev     = cats_arr[:, None, None] - exp_score[None, :, :]
            info    = (dev ** 2 * probs).sum(axis=0)
            info_df = pd.DataFrame(info, index=scores, columns=items)

            changes = ((exp_df.sum(axis=1) - used_scores)
                       / info_df.sum(axis=1)).clip(-1, 1)
            estimates -= changes
            iters += 1

        if warm_corr:
            pf = pd.DataFrame(1.0, columns=items, index=scores)
            estimates += self.warm(estimates, items, pf)

        self.abil_table = estimates

    def warm(self, abilities, items, person_filter):
        """
        Apply Warm's (1989) weighted maximum likelihood bias correction.

        Correction = (J1 - J2 + J3) / (2 * I^2) where:
            J1 = sum_i sum_k k^3 P(X_i=k)  (masked to observed items)
            J2 = 3 * (I + E^2) * E
            J3 = 2 * E^3
            I  = sum_i Var(X_i),  E = sum_i E[X_i]  (observed items only)
        The person_filter is critical: without it J1 includes unobserved items
        while J2/J3 exclude them, producing spuriously large corrections.

        Parameters
        ----------
        abilities : pandas.Series
            Current ability estimates, indexed by person.
        items : str or list
            Item subset.
        person_filter : pandas.DataFrame
            Binary mask (1.0 = responded, 0.0 = missing), shape (persons, items).

        Returns
        -------
        pandas.Series
            Warm bias correction terms to add to ML estimates.
        """
        if isinstance(items, str):
            items = [items]
        items = list(items)

        difficulties = self.diffs.loc[items]
        pf = person_filter.values if isinstance(person_filter, pd.DataFrame) else None

        probs, cats_arr = self._cat_probs_matrix(
            abilities.values, difficulties.values, self.thresholds
        )
        # probs: (K+1, N, I)

        exp_score = (cats_arr[:, None, None] * probs).sum(axis=0)    # (N, I)
        if pf is not None:
            exp_score *= pf

        dev  = cats_arr[:, None, None] - exp_score[None, :, :]       # (K+1, N, I)
        info = (dev ** 2 * probs).sum(axis=0)                         # (N, I)
        if pf is not None:
            info *= pf

        # part_1: Σ_i Σ_k k^3 P(X_i=k) -- must use MASKED probs
        # so unobserved items contribute 0, matching exp_score and info.
        cats3 = (cats_arr ** 3)[:, None, None]
        masked_probs = probs * pf[None, :, :] if pf is not None else probs
        part_1 = (cats3 * masked_probs).sum(axis=0).sum(axis=1)      # (N,)

        exp_sq = exp_score ** 2
        part_2 = 3 * ((info + exp_sq) * exp_score).sum(axis=1)       # (N,)
        part_3 = 2 * (exp_score ** 3).sum(axis=1)                    # (N,)

        info_sum = info.sum(axis=1)                                   # (N,)
        den = 2 * info_sum ** 2

        warm_correction = (part_1 - part_2 + part_3) / den
        return pd.Series(warm_correction, index=abilities.index)

    def csem(self, persons=None, abilities=None, items=None):
        """
        Compute the conditional standard error of measurement.

        CSEM = 1 / sqrt(I) where I is total Fisher information summed across
        observed items. Uses vectorised _cat_probs_matrix.

        Parameters
        ----------
        persons : list, str, or None, default None
            Person identifiers. If provided, overrides abilities.
        abilities : pandas.Series, float, list, or None, default None
            Ability estimates. If None, uses self.person_abilities.
        items : str, list, or None, default None
            Item subset. None uses all items.

        Returns
        -------
        numpy.ndarray
            CSEM values for each person/ability, in logits.
        """
        if abilities is None:
            abilities = self.person_abilities
        if isinstance(abilities, float):
            abilities = pd.Series({'Ability': abilities})
        if isinstance(abilities, list):
            abilities = pd.Series({f'Ability {a}': a for a in abilities})
        if persons is not None:
            abilities = self.person_abilities.loc[persons]

        if items is None or (isinstance(items, str) and items == 'all'):
            items = list(self.items)
        elif isinstance(items, str):
            items = [items]

        persons       = abilities.index
        difficulties  = self.diffs.loc[items]
        person_data   = self.dataframe.loc[persons, items]
        person_filter = person_data.notna().astype(float)

        probs, cats_arr = self._cat_probs_matrix(
            abilities.values, difficulties.values, self.thresholds
        )
        exp_score = (cats_arr[:, None, None] * probs).sum(axis=0)
        pf = person_filter.values
        exp_score *= pf

        dev  = cats_arr[:, None, None] - exp_score[None, :, :]
        info = (dev ** 2 * probs).sum(axis=0) * pf

        return 1.0 / (info.sum(axis=1) ** 0.5)

    # ------------------------------------------------------------------
    # Descriptive / count methods
    # ------------------------------------------------------------------

    def category_counts_item(self, item):
        """
        Return response frequency counts for a single item.

        Parameters
        ----------
        item : str
            Item identifier (must be a column in self.dataframe).

        Returns
        -------
        pandas.Series
            Count of each response category (0 to max_score), indexed by
            category value. Returns None and prints a message if not found.
        """

        if item not in self.dataframe.columns:
            warnings.warn(f'Invalid item name: {item!r}. Returning None.',
                          UserWarning, stacklevel=2)
            return None
        return (self.dataframe[item]
                .value_counts()
                .reindex(range(self.max_score + 1), fill_value=0)
                .astype(int))

    def category_counts_df(self):
        """
        Build and store a response frequency table across all items.

        All items share the same max_score in RSM, so there are no blank cells.
        Computes category counts (0 through max_score), total valid responses,
        and missing responses. Appends a Total row.

        Attributes set
        --------------
        category_counts : pandas.DataFrame
            Items as rows, categories plus Total and Missing as columns.
            A Total row is appended. All values are integers.
        """
        cat_counts = {item: self.category_counts_item(item)
                      for item in self.dataframe.columns}
        df = pd.DataFrame(cat_counts).T.sort_index(axis=1)
        df['Total']   = self.dataframe.count()
        df['Missing'] = self.no_of_persons - df['Total']
        df.loc['Total'] = df.sum()
        self.category_counts = df.astype(int)

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
                       method='cos',
                       constant=0.1,
                       matrix_power=3,
                       no_of_samples=100,
                       log_lik_tol=0.000001,
                       interval=None):
        """
        Compute all item, threshold, person, and test-level fit statistics.

        Auto-triggers calibrate(), std_errors(), and person_abils() if not yet
        run. Uses vectorised _cat_probs_matrix. Applies a cell-level guard
        (p > 0.9999) to prevent kurtosis/info^2 overflow in outfit statistics.
        Threshold fit statistics are computed by dichotomising at each threshold.

        Parameters
        ----------
        warm_corr : bool, default True
            Warm bias correction for ability estimates.
        se : bool, default True
            If True, computes bootstrap SEs. Required for test-level stats.
        test_stats : bool, default True
            If True, computes ISI, PSI, strata, and reliability.
        trim_cat_prob_dict : bool, default False
            If True, stores cat_prob_dict for non-extreme persons.
        tolerance : float, default 0.00001
            Newton-Raphson convergence tolerance.
        max_iters : int, default 100
            Maximum iterations.
        ext_score_adjustment : float, default 0.5
            Extreme score adjustment.
        method : str, default 'cos'
            Priority vector extraction method.
        constant : float, default 0.1
            Smoothing constant.
        matrix_power : int, default 3
            Matrix power for calibration.
        no_of_samples : int, default 100
            Bootstrap samples.
        log_lik_tol : float, default 0.000001
            Convergence tolerance for calibration.
        interval : float or None, default None
            CI width for bootstrap estimates.

        Attributes set
        --------------
        exp_score_df, info_df, kurtosis_df : pandas.DataFrame
            Expected scores, Fisher information, fourth moments. Degenerate
            cells (p > 0.9999) set to NaN.
        residual_df, std_residual_df : pandas.DataFrame
            Raw and standardised residuals.
        item_infit_ms, item_outfit_ms : pandas.Series
            Item infit and outfit mean-square.
        item_infit_zstd, item_outfit_zstd : pandas.Series
            Item infit and outfit Z statistics.
        item_facilities, response_counts : pandas.Series
            Item facilities and response counts.
        point_measure, exp_point_measure : pandas.Series
            Point-measure correlations.
        discrimination : pandas.Series
            Item discrimination indices.
        threshold_infit_ms, threshold_outfit_ms : pandas.Series
            Shared threshold infit and outfit mean-square.
        threshold_infit_zstd, threshold_outfit_zstd : pandas.Series
            Threshold Z statistics.
        threshold_point_measure, threshold_exp_point_measure : pandas.Series
            Threshold point-measure correlations.
        threshold_discrimination, threshold_rmsr : pandas.Series
            Threshold discrimination and RMSR.
        csem_vector, rsem_vector : pandas.Series
            Conditional and residual SEM per person.
        person_infit_ms, person_outfit_ms : pandas.Series
            Person infit and outfit mean-square.
        person_infit_zstd, person_outfit_zstd : pandas.Series
            Person Z statistics.
        isi, item_strata, item_reliability : float
            Item separation, strata, reliability (if test_stats).
        psi, person_strata, person_reliability : float
            Person separation, strata, reliability (if test_stats).
        """

        if not hasattr(self, 'thresholds'):
            self.calibrate(constant=constant, method=method,
                           matrix_power=matrix_power, log_lik_tol=log_lik_tol)
        if se and not hasattr(self, 'threshold_se'):
            self.std_errors(interval=interval, no_of_samples=no_of_samples,
                            constant=constant, method=method)
        if not hasattr(self, 'person_abilities'):
            self.person_abils(warm_corr=warm_corr, tolerance=tolerance,
                              max_iters=max_iters,
                              ext_score_adjustment=ext_score_adjustment)
        if not se:
            test_stats = False

        # Count valid responses per item and per person (before extreme filter)
        item_count   = self.dataframe.notna().sum(axis=0)
        person_count = self.dataframe.notna().sum(axis=1)

        df = self.dataframe.copy()
        scores     = df.sum(axis=1)
        max_scores = df.notna().sum(axis=1) * self.max_score
        df         = df[(scores > 0) & (scores < max_scores)]
        missing_mask = df.notna().astype(float)
        abilities    = self.person_abilities.loc[df.index]

        # Exclude persons with extreme ability estimates (diverged NR)
        abilities  = abilities[abilities.abs() <= 20]
        df         = df.loc[abilities.index]
        missing_mask = missing_mask.loc[abilities.index]

        diff_arr = self.diffs.values

        probs, cats_arr = self._cat_probs_matrix(
            abilities.values, diff_arr, self.thresholds
        )
        # probs: (K+1, N, I)

        # Cell-level guard: exclude person-item cells where one category
        # has near-certain probability (p > 0.9999, WINSTEPS convention).
        # Prevents kurtosis/info^2 overflow in outfit q-factor calculation.
        max_cat_prob = pd.DataFrame(
            probs.max(axis=0), index=abilities.index, columns=self.items
        )
        degenerate = max_cat_prob > 0.9999

        exp_score = (cats_arr[:, None, None] * probs).sum(axis=0)          # (N, I)
        self.exp_score_df = pd.DataFrame(exp_score,
                                         index=abilities.index,
                                         columns=self.items) * missing_mask
        self.exp_score_df[degenerate] = np.nan

        dev      = cats_arr[:, None, None] - exp_score[None, :, :]          # (K+1, N, I)
        info     = (dev ** 2 * probs).sum(axis=0)                            # (N, I)
        kurtosis = ((dev ** 4) * probs).sum(axis=0)                          # (N, I)

        self.info_df = pd.DataFrame(info, index=abilities.index,
                                    columns=self.items) * missing_mask
        self.info_df[degenerate] = np.nan

        self.kurtosis_df = pd.DataFrame(kurtosis, index=abilities.index,
                                        columns=self.items) * missing_mask
        self.kurtosis_df[degenerate] = np.nan

        self.residual_df     = self.dataframe.reindex(df.index) - self.exp_score_df
        self.std_residual_df = self.residual_df / (self.info_df ** 0.5)

        if trim_cat_prob_dict:
            # Store cat_prob_dict for downstream use if requested
            self.cat_prob_dict = {
                cat: pd.DataFrame(probs[cat], index=abilities.index,
                                  columns=self.items).loc[df.index]
                for cat in range(probs.shape[0])
            }

        # --- Item fit ---
        self.item_outfit_ms = (self.std_residual_df ** 2).mean()
        self.item_infit_ms  = (self.residual_df ** 2).sum() / self.info_df.sum()

        item_outfit_q = (((self.kurtosis_df / (self.info_df ** 2))
                          / (item_count ** 2)).sum() - (1 / item_count)) ** 0.5
        self.item_outfit_zstd = (((self.item_outfit_ms ** (1/3)) - 1)
                                 * (3 / item_outfit_q) + (item_outfit_q / 3))

        item_infit_q = ((self.kurtosis_df - self.info_df ** 2).sum()
                        / (self.info_df.sum() ** 2)) ** 0.5
        self.item_infit_zstd = (((self.item_infit_ms ** (1/3)) - 1)
                                * (3 / item_infit_q) + (item_infit_q / 3))

        self.response_counts  = self.dataframe.count(axis=0)
        self.item_facilities  = self.dataframe.mean(axis=0) / self.max_score

        (self.point_measure,
         self.exp_point_measure) = self.pt_meas(self.person_abilities,
                                                self.exp_score_df,
                                                self.info_df)

        # --- Threshold fit (dichotomised across all items) ---
        # For RSM, thresholds are shared so dich_thresh covers ALL items
        # for each threshold level, not per-item as in PCM.
        abil_df = pd.DataFrame(
            np.tile(self.person_abilities.values[:, None],
                    (1, self.no_of_items)),
            index=self.dataframe.index,
            columns=self.dataframe.columns
        )

        dich_thresh     = {}
        dich_thresh_exp = {}
        dich_thresh_var = {}
        dich_thresh_kur = {}
        dich_residuals  = {}
        dich_std_residuals = {}

        for t in range(self.max_score):
            # Dichotomise: keep only persons scoring t or t+1, recode as 0/1
            dich = self.dataframe.where(
                self.dataframe.isin([t, t + 1]), np.nan
            ) - t
            dich_thresh[t + 1] = dich

            mm = dich.notna().astype(float).replace(0, np.nan)

            # Threshold location for threshold t+1: diff_i + tau_{t+1}
            # diff_df[n,i] = delta_i + tau_{t+1}  (identical across persons,
            # so tile the (1, I) row vector to (N, I) before constructing DataFrame)
            diff_df = pd.DataFrame(
                np.tile(self.diffs.values + self.thresholds[t + 1],
                        (len(self.dataframe.index), 1)),
                index=self.dataframe.index,
                columns=self.dataframe.columns
            )

            p = 1.0 / (1.0 + np.exp(diff_df - abil_df))
            p_masked = p * mm

            dich_thresh_exp[t + 1] = p_masked
            dich_thresh_var[t + 1] = p_masked * (1 - p_masked) * mm
            dich_thresh_kur[t + 1] = (
                ((-p_masked) ** 4) * (1 - p_masked) +
                ((1 - p_masked) ** 4) * p_masked
            ) * mm
            dich_residuals[t + 1]     = dich - p_masked
            dich_std_residuals[t + 1] = dich_residuals[t + 1] / (dich_thresh_var[t + 1] ** 0.5)

        dich_thresh_count = {
            t + 1: dich_thresh[t + 1].count().sum()
            for t in range(self.max_score)
        }

        self.threshold_outfit_ms = pd.Series({
            t + 1: ((dich_std_residuals[t + 1] ** 2).sum().sum()
                    / dich_thresh_count[t + 1]
                    if dich_thresh_count[t + 1] > 0 else np.nan)
            for t in range(self.max_score)
        })

        self.threshold_infit_ms = pd.Series({
            t + 1: ((dich_residuals[t + 1] ** 2).sum().sum()
                    / dich_thresh_var[t + 1].sum().sum()
                    if dich_thresh_var[t + 1].sum().sum() > 0 else np.nan)
            for t in range(self.max_score)
        })

        threshold_outfit_q = pd.Series({
            t + 1: (((dich_thresh_kur[t + 1]
                      / (dich_thresh_var[t + 1] ** 2))
                     / (dich_thresh_count[t + 1] ** 2)).sum().sum()
                    - (1 / dich_thresh_count[t + 1])
                    if dich_thresh_count[t + 1] > 0 else np.nan)
            for t in range(self.max_score)
        }) ** 0.5

        self.threshold_outfit_zstd = (((self.threshold_outfit_ms ** (1/3)) - 1)
                                      * (3 / threshold_outfit_q)
                                      + (threshold_outfit_q / 3))

        threshold_infit_q = pd.Series({
            t + 1: ((dich_thresh_kur[t + 1]
                     - dich_thresh_var[t + 1] ** 2).sum().sum()
                    / (dich_thresh_var[t + 1].sum().sum() ** 2)
                    if dich_thresh_var[t + 1].sum().sum() > 0 else np.nan)
            for t in range(self.max_score)
        }) ** 0.5

        self.threshold_infit_zstd = (((self.threshold_infit_ms ** (1/3)) - 1)
                                     * (3 / threshold_infit_q)
                                     + (threshold_infit_q / 3))

        abil_deviation = self.person_abilities - self.person_abilities.mean()

        # Threshold point-measure correlations
        pm_num = pd.Series({
            t + 1: ((dich_thresh[t + 1] - dich_thresh[t + 1].mean())
                    .mul(abil_deviation, axis=0).sum().sum()
                    if dich_thresh[t + 1].count().sum() > 0 else np.nan)
            for t in range(self.max_score)
        })
        pm_den = pd.Series({
            t + 1: (((dich_thresh[t + 1] - dich_thresh[t + 1].mean()) ** 2)
                    .sum().sum()
                    * (abil_deviation ** 2).sum()) ** 0.5
            for t in range(self.max_score)
        })
        self.threshold_point_measure = pm_num / pm_den

        exp_pm_dict = {
            t + 1: dich_thresh_exp[t + 1] - dich_thresh_exp[t + 1].mean()
            for t in range(self.max_score)
        }
        exp_pm_num = pd.Series({
            t + 1: exp_pm_dict[t + 1].mul(abil_deviation, axis=0).sum().sum()
            for t in range(self.max_score)
        })
        exp_pm_den = pd.Series({
            t + 1: ((exp_pm_dict[t + 1] ** 2)
                    + dich_thresh_var[t + 1]).sum().sum()
            for t in range(self.max_score)
        })
        exp_pm_den *= (abil_deviation ** 2).sum()
        exp_pm_den  = exp_pm_den ** 0.5
        self.threshold_exp_point_measure = exp_pm_num / exp_pm_den

        self.threshold_rmsr = pd.Series({
            t + 1: ((dich_residuals[t + 1] ** 2).sum().sum()
                    / dich_residuals[t + 1].count().sum()) ** 0.5
            if dich_residuals[t + 1].count().sum() > 0 else np.nan
            for t in range(self.max_score)
        })

        # Threshold discrimination
        differences = {
            t + 1: pd.DataFrame(
                self.person_abilities.values[:, None]
                - (self.diffs.values[None, :] + self.thresholds[t + 1]),
                index=self.dataframe.index,
                columns=self.dataframe.columns
            )
            for t in range(self.max_score)
        }
        disc_num = pd.Series({
            t + 1: (differences[t + 1] * dich_residuals[t + 1]).sum().sum()
            for t in range(self.max_score)
        })
        disc_den = pd.Series({
            t + 1: (dich_thresh_var[t + 1] * differences[t + 1] ** 2).sum().sum()
            for t in range(self.max_score)
        })
        self.threshold_discrimination = 1 + disc_num / disc_den

        # --- Person fit ---
        self.csem_vector = 1.0 / (self.info_df.sum(axis=1) ** 0.5)
        self.rsem_vector = ((self.residual_df ** 2).sum(axis=1) ** 0.5) / self.info_df.sum(axis=1)

        self.person_outfit_ms = (self.std_residual_df ** 2).mean(axis=1)
        self.person_outfit_ms.name = 'Outfit MS'
        self.person_infit_ms  = ((self.residual_df ** 2).sum(axis=1)
                                  / self.info_df.sum(axis=1))
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
            self.isi = (self.diffs.var() / (self.item_se ** 2).mean() - 1) ** 0.5
            self.item_strata      = (4 * self.isi + 1) / 3
            self.item_reliability = self.isi ** 2 / (1 + self.isi ** 2)

            # BUG FIX: original RSM formula was:
            #   (var^0.5 - mean_rsem2) / mean_rsem2^0.5   <- wrong: sqrt taken early
            # Correct Wright & Masters formula:
            #   sqrt((var - mean_rsem2) / mean_rsem2)
            mean_rsem2 = (self.rsem_vector ** 2).mean()
            self.psi = ((np.var(self.person_abilities) - mean_rsem2)
                        / mean_rsem2) ** 0.5
            self.person_strata      = (4 * self.psi + 1) / 3
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
            Convergence tolerance.
        max_iters : int, default 100
            Maximum iterations.
        ext_score_adjustment : float, default 0.5
            Extreme score adjustment.
        constant : float, default 0.1
            Smoothing constant.
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
            warnings.warn('PCA of standardised residuals failed. '
                          'Eigenvectors and loadings set to None.',
                          UserWarning, stacklevel=2)
            self.eigenvectors = self.eigenvalues = None
            self.variance_explained = self.loadings = None

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
        """
        Build and store the item statistics summary table.

        Auto-triggers std_errors() and fit_statistics() if not yet run.

        Parameters
        ----------
        full : bool, default False
            If True, sets zstd=True, point_measure_corr=True, interval=0.95.
        zstd : bool, default False
            If True, includes Infit Z and Outfit Z columns.
        point_measure_corr : bool, default False
            If True, includes point-measure correlation columns.
        dp : int, default 3
            Decimal places.
        warm_corr : bool, default True
            Warm bias correction.
        tolerance : float, default 0.00001
            Newton-Raphson convergence tolerance.
        max_iters : int, default 100
            Maximum iterations.
        ext_score_adjustment : float, default 0.5
            Extreme score adjustment.
        method : str, default 'cos'
            Priority vector extraction method.
        constant : float, default 0.1
            Smoothing constant.
        no_of_samples : int, default 100
            Bootstrap samples.
        interval : float or None, default None
            CI width; if provided, percentile bound columns included.

        Attributes set
        --------------
        item_stats : pandas.DataFrame
            Item statistics with items as rows. Always contains Estimate,
            SE, Count, Facility, Infit MS, Outfit MS.
        """

        if full:
            zstd = True
            point_measure_corr = True
            if interval is None:
                interval = 0.95

        if not hasattr(self, 'threshold_se') or (self.threshold_low is None and interval is not None):
            self.std_errors(interval=interval, no_of_samples=no_of_samples,
                            constant=constant, method=method)
        if not hasattr(self, 'item_infit_ms'):
            self.fit_statistics(warm_corr=warm_corr, tolerance=tolerance,
                                max_iters=max_iters,
                                ext_score_adjustment=ext_score_adjustment,
                                method=method, constant=constant,
                                no_of_samples=no_of_samples, interval=interval)

        stats = pd.DataFrame(index=self.dataframe.columns)
        stats['Estimate'] = self.diffs.round(dp)
        stats['SE']       = self.item_se.round(dp)
        if interval is not None:
            stats[f'{round((1 - interval) * 50, 1)}%'] = self.item_low.round(dp)
            stats[f'{round((1 + interval) * 50, 1)}%'] = self.item_high.round(dp)
        stats['Count']    = self.response_counts.astype(int)
        stats['Facility'] = self.item_facilities.round(dp)
        stats['Infit MS'] = self.item_infit_ms.round(dp)
        if zstd:
            stats['Infit Z'] = self.item_infit_zstd.round(dp)
        stats['Outfit MS'] = self.item_outfit_ms.round(dp)
        if zstd:
            stats['Outfit Z'] = self.item_outfit_zstd.round(dp)
        if point_measure_corr:
            stats['PM corr']     = self.point_measure.round(dp)
            stats['Exp PM corr'] = self.exp_point_measure.round(dp)
        self.item_stats = stats

    def threshold_stats_df(self,
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
                           no_of_samples=100,
                           interval=None):
        """
        Build and store the threshold statistics summary table.

        Auto-triggers fit_statistics() if not yet run. Reports statistics for
        the max_score shared Rasch-Andrich thresholds (thresholds[1..max_score]).
        The tau_0=0 sentinel is excluded. Unlike PCM, RSM has one shared
        threshold set across all items.

        Parameters
        ----------
        full : bool, default False
            If True, sets zstd=True, disc=True, point_measure_corr=True, interval=0.95.
        zstd : bool, default False
            If True, includes Infit Z and Outfit Z columns.
        disc : bool, default False
            If True, includes threshold discrimination column.
        point_measure_corr : bool, default False
            If True, includes point-measure correlation columns.
        dp : int, default 3
            Decimal places.
        warm_corr : bool, default True
            Warm bias correction.
        tolerance : float, default 0.00001
            Convergence tolerance.
        max_iters : int, default 100
            Maximum iterations.
        ext_score_adjustment : float, default 0.5
            Extreme score adjustment.
        method : str, default 'cos'
            Priority vector extraction method.
        constant : float, default 0.1
            Smoothing constant.
        no_of_samples : int, default 100
            Bootstrap samples.
        interval : float or None, default None
            CI width.

        Attributes set
        --------------
        threshold_stats : pandas.DataFrame
            Threshold statistics, rows Threshold 1..Threshold max_score.
            Always contains Estimate, SE, Infit MS, Outfit MS.
        """

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

        # thresholds[1..max_score] are the actual threshold estimates;
        # thresholds[0] = 0 is the tau_0 sentinel and is not reported.
        idx = [f'Threshold {t + 1}' for t in range(self.max_score)]
        stats = pd.DataFrame(index=idx)
        stats['Estimate'] = self.thresholds[1:].round(dp)
        stats['SE']       = self.threshold_se[1:].round(dp)
        if interval is not None:
            stats[f'{round((1 - interval) * 50, 1)}%'] = self.threshold_low[1:].round(dp)
            stats[f'{round((1 + interval) * 50, 1)}%'] = self.threshold_high[1:].round(dp)
        stats['Infit MS']  = self.threshold_infit_ms.values.round(dp)
        if zstd:
            stats['Infit Z']  = self.threshold_infit_zstd.values.round(dp)
        stats['Outfit MS'] = self.threshold_outfit_ms.values.round(dp)
        if zstd:
            stats['Outfit Z'] = self.threshold_outfit_zstd.values.round(dp)
        if disc:
            stats['Discrim'] = self.threshold_discrimination.values.round(dp)
        if point_measure_corr:
            stats['PM corr']     = self.threshold_point_measure.values.round(dp)
            stats['Exp PM corr'] = self.threshold_exp_point_measure.values.round(dp)
        self.threshold_stats = stats

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

        Auto-triggers fit_statistics() if not yet run.

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
            Convergence tolerance.
        max_iters : int, default 100
            Maximum iterations.
        ext_score_adjustment : float, default 0.5
            Extreme score adjustment.
        method : str, default 'cos'
            Priority vector extraction method.
        constant : float, default 0.1
            Smoothing constant.

        Attributes set
        --------------
        person_stats : pandas.DataFrame
            Person statistics with persons as rows. Contains Estimate, CSEM,
            Score, Max score, p, Infit MS, Infit Z, Outfit MS, Outfit Z.
            Optional: RSEM.
        """

        if not hasattr(self, 'person_infit_ms'):
            self.fit_statistics(warm_corr=warm_corr, tolerance=tolerance,
                                max_iters=max_iters,
                                ext_score_adjustment=ext_score_adjustment,
                                method=method, constant=constant)
        if full:
            rsem = True

        idx   = self.dataframe.index
        stats = pd.DataFrame(index=idx)
        stats['Estimate']  = self.person_abilities.round(dp)
        stats['CSEM']      = self.csem_vector.round(dp)
        if rsem:
            stats['RSEM']  = self.rsem_vector.round(dp)
        stats['Score']     = self.dataframe.sum(axis=1).astype(int)
        stats['Max score'] = (self.dataframe.count(axis=1) * self.max_score).astype(int)
        stats['p']         = (self.dataframe.mean(axis=1) / self.max_score).round(dp)

        # BUG FIX: original used .update(dict) which ignores index alignment.
        for col, src in [('Infit MS',  self.person_infit_ms),
                         ('Infit Z',   self.person_infit_zstd),
                         ('Outfit MS', self.person_outfit_ms),
                         ('Outfit Z',  self.person_outfit_zstd)]:
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
        """
        Build and store the test-level summary statistics table.

        Auto-triggers fit_statistics() if not yet run. Produces a two-column
        table (Items, Persons). RSM has no threshold separation row because
        thresholds are shared across items.

        Parameters
        ----------
        dp : int, default 3
            Decimal places.
        warm_corr : bool, default True
            Warm bias correction.
        tolerance : float, default 0.00001
            Convergence tolerance.
        max_iters : int, default 100
            Maximum iterations.
        ext_score_adjustment : float, default 0.5
            Extreme score adjustment.
        method : str, default 'cos'
            Priority vector extraction method.
        constant : float, default 0.1
            Smoothing constant.

        Attributes set
        --------------
        test_stats : pandas.DataFrame
            Two-column table (Items, Persons) with rows:
            Mean, SD, Separation ratio, Strata, Reliability.
        """

        if not hasattr(self, 'psi'):
            self.fit_statistics(warm_corr=warm_corr, tolerance=tolerance,
                                max_iters=max_iters,
                                ext_score_adjustment=ext_score_adjustment,
                                method=method, constant=constant)

        # RSM test stats have no threshold separation row (thresholds are
        # shared, not item-specific, so threshold ISI is not meaningful here).
        self.test_stats = pd.DataFrame({
            'Items':   [self.diffs.mean(), self.diffs.std(),
                        self.isi, self.item_strata, self.item_reliability],
            'Persons': [self.person_abilities.mean(), self.person_abilities.std(),
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
        """
        Export item, threshold, person, and test statistics to file.

        Parameters
        ----------
        filename : str
            Output filename or path.
        format : str, default 'csv'
            'csv' saves four separate CSV files. 'xlsx' saves to a single workbook.
        dp : int, default 3
            Decimal places.
        warm_corr : bool, default True
            Warm bias correction.
        tolerance : float, default 0.00001
            Convergence tolerance.
        max_iters : int, default 100
            Maximum iterations.
        ext_score_adjustment : float, default 0.5
            Extreme score adjustment.
        method : str, default 'cos'
            Priority vector extraction method.
        constant : float, default 0.1
            Smoothing constant.
        no_of_samples : int, default 100
            Bootstrap samples.
        interval : float or None, default None
            CI width.
        """

        for attr, method_name, kwargs in [
            ('item_stats',      'item_stats_df',
             dict(dp=dp, warm_corr=warm_corr, tolerance=tolerance,
                  max_iters=max_iters, ext_score_adjustment=ext_score_adjustment,
                  method=method, constant=constant,
                  no_of_samples=no_of_samples, interval=interval)),
            ('threshold_stats', 'threshold_stats_df',
             dict(dp=dp, warm_corr=warm_corr, tolerance=tolerance,
                  max_iters=max_iters, ext_score_adjustment=ext_score_adjustment,
                  method=method, constant=constant,
                  no_of_samples=no_of_samples, interval=interval)),
            ('person_stats',    'person_stats_df',
             dict(dp=dp, warm_corr=warm_corr, tolerance=tolerance,
                  max_iters=max_iters, ext_score_adjustment=ext_score_adjustment,
                  method=method, constant=constant)),
            ('test_stats',      'test_stats_df',
             dict(dp=dp, warm_corr=warm_corr, tolerance=tolerance,
                  max_iters=max_iters, ext_score_adjustment=ext_score_adjustment,
                  method=method, constant=constant)),
        ]:
            if not hasattr(self, attr):
                getattr(self, method_name)(**kwargs)

        if format == 'xlsx':
            if not filename.endswith('.xlsx'):
                filename += '.xlsx'
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                self.item_stats.to_excel(writer, sheet_name='Item statistics')
                self.threshold_stats.to_excel(writer, sheet_name='Threshold statistics')
                self.person_stats.to_excel(writer, sheet_name='Person statistics')
                self.test_stats.to_excel(writer, sheet_name='Test statistics')
        else:
            if filename.endswith('.csv'):
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
                       tolerance=0.00001,
                       max_iters=100,
                       ext_score_adjustment=0.5,
                       method='cos',
                       constant=0.1):
        """
        Export residual correlation analysis results to file.

        Parameters
        ----------
        filename : str
            Output filename or path.
        format : str, default 'csv'
            'csv' or 'xlsx'.
        single : bool, default True
            If True, writes all tables to a single file/sheet.
        dp : int, default 3
            Decimal places.
        warm_corr : bool, default True
            Warm bias correction.
        tolerance : float, default 0.00001
            Convergence tolerance.
        max_iters : int, default 100
            Maximum iterations.
        ext_score_adjustment : float, default 0.5
            Extreme score adjustment.
        method : str, default 'cos'
            Priority vector extraction method.
        constant : float, default 0.1
            Smoothing constant.
        """

        if not hasattr(self, 'eigenvectors'):
            # BUG FIX: must call res_corr_analysis (not just fit_statistics) to set eigenvectors
            self.res_corr_analysis(warm_corr=warm_corr, tolerance=tolerance,
                                   max_iters=max_iters,
                                   ext_score_adjustment=ext_score_adjustment,
                                   method=method, constant=constant)

        frames       = [self.eigenvectors, self.eigenvalues,
                        self.variance_explained, self.loadings]
        sheet_single = 'Item residual analysis'
        sheet_multi  = ['Eigenvectors', 'Eigenvalues',
                        'Variance explained', 'Principal Component loadings']
        csv_suffixes = ['_eigenvectors', '_eigenvalues',
                        '_variance_explained', '_principal_component_loadings']

        if format == 'xlsx':
            if not filename.endswith('.xlsx'):
                filename += '.xlsx'
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                if single:
                    row = 0
                    for frame in frames:
                        frame.round(dp).to_excel(writer, sheet_name=sheet_single,
                                                 startrow=row, startcol=0)
                        row += frame.shape[0] + 2
                else:
                    for frame, sheet in zip(frames, sheet_multi):
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

    def class_intervals(self, items=None, no_of_classes=5):
        """
        Compute class interval mean abilities and mean observed total scores.

        Partitions persons into quantile-based ability groups and computes
        mean ability and mean observed total score within each group.
        Used for observed-data overlays on TCC and ICC plots.
        Requires self.person_abilities to exist.

        Parameters
        ----------
        items : str, list, or None, default None
            Item subset. None uses all items.
        no_of_classes : int, default 5
            Number of class intervals.

        Returns
        -------
        mean_abilities : pandas.Series
            Mean ability within each class interval.
        obs : pandas.Series
            Mean observed total score within each class interval.
        """

        if isinstance(items, str) and items in ('all', 'none'):
            items = None
        if items is None:
            items = self.dataframe.columns.tolist()

        class_groups = [f'class_{i + 1}' for i in range(no_of_classes)]
        df    = self.dataframe[items].dropna(how='any')
        abils = self.person_abilities.loc[df.index]
        q     = abils.quantile([(i + 1) / no_of_classes
                                for i in range(no_of_classes - 1)])

        mask_dict = {
            'class_1': abils < q.values[0],
            f'class_{no_of_classes}': abils >= q.values[-1],
            **{f'class_{i + 2}': ((abils >= q.values[i]) &
                                   (abils < q.values[i + 1]))
               for i in range(no_of_classes - 2)}
        }
        mean_abilities = pd.Series({cg: abils[mask_dict[cg]].mean()
                                    for cg in class_groups})
        obs = pd.concat({cg: pd.Series(df[mask_dict[cg]].mean().sum())
                         for cg in class_groups})
        return mean_abilities, obs

    def class_intervals_cats(self, abilities, item=None, no_of_classes=5):
        """
        Compute class interval mean abilities and observed category proportions.

        Partitions persons into quantile-based ability groups and computes the
        proportion of each response category within each group. When item=None,
        pools across all items using ability relative to each item's difficulty.
        Used for observed-data overlays on CRC plots.

        Parameters
        ----------
        abilities : pandas.Series
            Person ability estimates indexed by person identifier.
        item : str or None, default None
            Item identifier. If None, pools across all items.
        no_of_classes : int, default 5
            Number of class intervals.

        Returns
        -------
        mean_abilities : pandas.Series
            Mean ability within each class interval.
        obs_props : numpy.ndarray
            Shape (no_of_classes, max_score+1) with proportions of each
            response category in each class interval.
        """

        class_groups = [f'class_{i + 1}' for i in range(no_of_classes)]
        df = self.dataframe.copy()

        if item is None:
            # Use ability relative to each item's difficulty
            abil_df = pd.DataFrame(
                {item_: abilities - self.diffs[item_]
                 for item_ in self.dataframe.columns}
            ) * df.notna().astype(float).replace(0, np.nan)
            mask_scores = df.unstack()
            mask_abils  = abil_df.unstack()
        else:
            mask_scores = df[item].dropna()
            mask_abils  = abilities.loc[df.index]

        q = mask_abils.quantile([(i + 1) / no_of_classes
                                  for i in range(no_of_classes - 1)])
        mask_dict = {
            'class_1': mask_abils < q.values[0],
            f'class_{no_of_classes}': mask_abils >= q.values[-1],
            **{f'class_{i + 2}': ((mask_abils >= q.values[i]) &
                                   (mask_abils < q.values[i + 1]))
               for i in range(no_of_classes - 2)}
        }
        mean_abilities = pd.Series({cg: mask_abils[mask_dict[cg]].mean()
                                    for cg in class_groups})
        obs_props = np.array([
            [(mask_scores[mask_dict[cg]] == cat).sum()
             for cat in range(self.max_score + 1)]
            for cg in class_groups
        ], dtype=float)
        obs_props /= obs_props.sum(axis=1, keepdims=True)
        return mean_abilities, obs_props

    def class_intervals_thresholds(self, item=None, no_of_classes=5):
        """
        Compute class interval data for threshold characteristic curves.

        For each threshold (adjacent category pair), dichotomises responses,
        partitions persons into quantile-based ability groups, and computes the
        mean ability and observed proportion in the higher category within each
        group. When item=None, pools across all items.
        Auto-triggers person_abils() if not yet run.

        Parameters
        ----------
        item : str or None, default None
            Item identifier. If None, pools across all items.
        no_of_classes : int, default 5
            Number of class intervals.

        Returns
        -------
        mean_abilities : numpy.ndarray
            Shape (no_of_classes, max_score).
        obs_props : numpy.ndarray
            Shape (no_of_classes, max_score).
        """

        if not hasattr(self, 'person_abilities'):
            self.person_abils(warm_corr=False)

        class_groups = [f'class_{i + 1}' for i in range(no_of_classes)]
        df = self.dataframe.copy()

        # Build ability DataFrame; subtract item difficulty if not item-specific
        abil_df = pd.DataFrame(
            {it: self.person_abilities for it in self.dataframe.columns}
        )
        if item is None:
            for it in self.dataframe.columns:
                abil_df[it] -= self.diffs[it]
        else:
            df     = df[item]
            abil_df = abil_df[item]

        def make_masks(abils):
            q = abils.quantile([(i + 1) / no_of_classes
                                for i in range(no_of_classes - 1)])
            md = {
                'class_1': abils < q.values[0],
                f'class_{no_of_classes}': abils >= q.values[-1],
                **{f'class_{i + 2}': ((abils >= q.values[i]) &
                                       (abils < q.values[i + 1]))
                   for i in range(no_of_classes - 2)}
            }
            return {cg: md[cg][md[cg]].index for cg in class_groups}

        mean_abilities, obs_props = [], []
        for t in range(self.max_score):
            cond_df   = df[df.isin([t, t + 1])] - t
            cond_mask = cond_df.notna().astype(float).replace(0, np.nan)
            cond_abils = abil_df * cond_mask

            if item is None:
                obs_df = pd.DataFrame({
                    'ability': cond_abils.stack(),
                    'score':   cond_df.stack()
                }).droplevel(level=1)
            else:
                obs_df = pd.DataFrame({'ability': cond_abils, 'score': cond_df})

            masks = make_masks(obs_df['ability'])
            mean_abilities.append([obs_df.loc[masks[cg]]['ability'].mean()
                                   for cg in class_groups])
            obs_props.append([obs_df.loc[masks[cg]]['score'].mean()
                              for cg in class_groups])

        return np.array(mean_abilities).T, np.array(obs_props).T

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    def plot_data(self,
                  x_data,
                  y_data,
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
                  x_min=-5,
                  x_max=5,
                  y_max=0,
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
        Core plotting engine for all RSM item and test characteristic curves.

        Renders curves against an ability x-axis with optional observed overlays,
        threshold lines, central difference lines, score lines, information lines,
        and CSEM lines. Called internally by icc(), crcs(), threshold_ccs(),
        iic(), tcc(), test_info(), and test_csem().

        Parameters
        ----------
        x_data : array-like
            X-axis values (typically ability grid -20 to 20).
        y_data : numpy.ndarray
            2-D array shape (len(x_data), n_curves).
        items : str, list, or None
            Item(s) being plotted.
        obs : bool, list, or None
            Controls observed data overlay.
        x_obs_data, y_obs_data : array-like
            Observed data point coordinates.
        thresh_lines : bool, default False
            Draw vertical lines at absolute threshold locations.
        central_diff : bool, default False
            Draw a line at the item central difficulty.
        score_lines_item : list, default [None, None]
            [item_name, list_of_scores] for item-level score lines.
        score_lines_test : list or None
            Raw total scores for test-level score reference lines.
        point_info_lines_item : list, default [None, None]
            Item-level information reference lines.
        point_info_lines_test : list or None
            Test-level information reference lines.
        point_csem_lines : list or None
            CSEM reference lines.
        score_labels : bool, default False
            Annotate intersections with values.
        x_min, x_max : float
            Displayed x-axis limits.
        y_max : float, default 0
            Upper y-axis limit. If <= 0, auto-scaled.
        warm : bool, default True
            Used for score line ability lookups.
        cat_highlight : int or None
            Category to shade blue.
        graph_title, y_label : str
            Plot title and y-axis label.
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
        title_font_size, axis_font_size, labelsize : int
            Font sizes.
        tex : bool, default True
            Attempt LaTeX rendering.
        plot_density : int, default 300
            Output DPI.
        filename : str or None
            If provided, saves the plot.
        file_format : str, default 'png'
            Output file format.

        Returns
        -------
        matplotlib.figure.Figure
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
                    n_obs = y_obs_data.shape[1]
                    x_is_series = isinstance(x_obs_data, pd.Series)
                    for j in range(n_obs):
                        col = (scalarMap.to_rgba(j) if 'multi' not in palette
                               else color_map[j])
                        xd = x_obs_data if x_is_series else x_obs_data[:, j]
                        ax.plot(xd, y_obs_data[:, j], 'o', color=col)
                except Exception:
                    pass

            if thresh_lines:
                for t in range(self.max_score):
                    xval = (self.thresholds[t + 1] if items is None
                            else self.thresholds[t + 1] + self.diffs.loc[items])
                    ax.axvline(x=xval, color='black', linestyle='--')

            if central_diff:
                xval = 0 if items is None else self.diffs.loc[items]
                ax.axvline(x=xval, color='darkred', linestyle='--')

            if score_lines_item[1] is not None:
                item = score_lines_item[0]
                if (all(s > 0 for s in score_lines_item[1]) and
                        all(s < self.max_score for s in score_lines_item[1])):
                    for s in score_lines_item[1]:
                        abil = self.score_abil(s, items=[item], warm_corr=False)
                        ax.vlines(x=abil, ymin=-100, ymax=s, color='black', linestyles='dashed')
                        ax.hlines(y=s, xmin=-100, xmax=abil, color='black', linestyles='dashed')
                        if score_labels:
                            ax.text(abil + (x_max - x_min) / 100, y_max / 50, str(round(abil, 2)))
                            ax.text(x_min + (x_max - x_min) / 100, s + y_max / 50, str(s))
                else:
                    warnings.warn('Invalid score for score line: value must be '
                                  'strictly between 0 and the item maximum score.',
                                  UserWarning, stacklevel=2)

            if score_lines_test is not None:
                item_keys = (self.dataframe.columns if items is None
                             else ([items] if isinstance(items, str) else items))
                n_items = len(item_keys)
                if (all(s > 0 for s in score_lines_test) and
                        all(s < self.max_score * n_items for s in score_lines_test)):
                    for s in score_lines_test:
                        abil = self.score_abil(s, items=list(item_keys), warm_corr=warm)
                        ax.vlines(x=abil, ymin=-100, ymax=s, color='black', linestyles='dashed')
                        ax.hlines(y=s, xmin=-100, xmax=abil, color='black', linestyles='dashed')
                        if score_labels:
                            ax.text(abil + (x_max - x_min) / 100, y_max / 50, str(round(abil, 2)))
                            ax.text(x_min + (x_max - x_min) / 100, s + y_max / 50, str(s))
                else:
                    warnings.warn('Invalid score for score line: value must be '
                                  'strictly between 0 and the test maximum score.',
                                  UserWarning, stacklevel=2)

            if point_info_lines_item[1] is not None:
                item = point_info_lines_item[0]
                for abil in point_info_lines_item[1]:
                    info = self.variance(abil, self.diffs[item], self.thresholds)
                    ax.vlines(x=abil, ymin=-100, ymax=info, color='black', linestyles='dashed')
                    ax.hlines(y=info, xmin=-100, xmax=abil, color='black', linestyles='dashed')
                    if score_labels:
                        ax.text(abil + (x_max - x_min) / 100, y_max / 50, str(round(abil, 2)))
                        ax.text(x_min + (x_max - x_min) / 100, info + y_max / 50, str(round(info, 3)))

            if point_info_lines_test is not None:
                item_keys = (self.dataframe.columns if items is None else items)
                for abil in point_info_lines_test:
                    info = sum(self.variance(abil, self.diffs[it], self.thresholds)
                               for it in item_keys)
                    ax.vlines(x=abil, ymin=-100, ymax=info, color='black', linestyles='dashed')
                    ax.hlines(y=info, xmin=-100, xmax=abil, color='black', linestyles='dashed')
                    if score_labels:
                        ax.text(abil + (x_max - x_min) / 100, y_max / 50, str(round(abil, 2)))
                        ax.text(x_min + (x_max - x_min) / 100, info + y_max / 50, str(round(info, 3)))

            if point_csem_lines is not None:
                item_keys = (self.dataframe.columns if items is None else items)
                for abil in point_csem_lines:
                    info = sum(self.variance(abil, self.diffs[it], self.thresholds)
                               for it in item_keys)
                    csem = 1.0 / (info ** 0.5)
                    ax.vlines(x=abil, ymin=-100, ymax=csem, color='black', linestyles='dashed')
                    ax.hlines(y=csem, xmin=-100, xmax=abil, color='black', linestyles='dashed')
                    if score_labels:
                        ax.text(abil + (x_max - x_min) / 100, y_max / 50, str(round(abil, 2)))
                        ax.text(x_min + (x_max - x_min) / 100, csem + y_max / 50, str(round(csem, 3)))

            if items is not None and cat_highlight in range(self.max_score + 1):
                if cat_highlight == 0:
                    ax.axvspan(-100, self.diffs[items] + self.thresholds[1],
                               facecolor='blue', alpha=0.2)
                elif cat_highlight == self.max_score:
                    ax.axvspan(self.diffs[items] + self.thresholds[self.max_score],
                               100, facecolor='blue', alpha=0.2)
                else:
                    lo = self.diffs[items] + self.thresholds[cat_highlight]
                    hi = self.diffs[items] + self.thresholds[cat_highlight + 1]
                    if hi > lo:
                        ax.axvspan(lo, hi, facecolor='blue', alpha=0.2)

            if y_max <= 0:
                y_max = float(y_data.max()) * 1.1

            ax.set_xlim(x_min, x_max)
            ax.set_ylim(0, y_max)
            ax.set_xlabel('Ability', fontsize=axis_font_size, fontweight='bold')
            ax.set_ylabel(y_label, fontsize=axis_font_size, fontweight='bold')
            ax.set_title(graph_title, fontsize=title_font_size, fontweight='bold')
            ax.grid(True)
            ax.tick_params(axis='x', labelsize=labelsize)
            ax.tick_params(axis='y', labelsize=labelsize)

            if filename is not None:
                graph.savefig(f'{filename}.{file_format}', dpi=plot_density)

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

        Displays modelled expected score as a function of ability. Optionally
        overlays observed class-interval mean scores.

        Parameters
        ----------
        item : str
            Item identifier.
        obs : bool, default False
            If True, overlays observed class-interval mean scores.
        no_of_classes : int, default 5
            Number of class intervals.
        title : str or None, default None
            Plot title.
        thresh_lines : bool, default False
            Draw vertical lines at absolute threshold locations (tau_k + delta_i).
        central_diff : bool, default False
            Draw a line at the item central difficulty.
        score_lines : list or None, default None
            Raw scores at which to draw reference lines.
        score_labels : bool, default False
            Annotate score line intersections.
        cat_highlight : int or None, default None
            Category to shade.
        xmin, xmax : float
            Ability axis limits.
        plot_style, palette, black, font : see plot_data().
        title_font_size, axis_font_size, labelsize : int
            Font sizes.
        filename : str or None
            If provided, saves the plot.
        file_format : str, default 'png'
            Output format.
        dpi : int, default 300
            Resolution.

        Returns
        -------
        matplotlib.figure.Figure
        """
        # BUG FIX: typo 'person_abiliites' -> 'person_abilities'
        if obs and not hasattr(self, 'person_abilities'):
            self.person_abils(warm_corr=False)

        xobsdata = yobsdata = np.array(np.nan)
        if obs:
            mean_abilities, obs_means = self.class_intervals(
                items=item, no_of_classes=no_of_classes
            )
            xobsdata = pd.Series(mean_abilities)
            yobsdata = np.array(obs_means).reshape(-1, 1)

        abilities = np.arange(-20, 20, 0.1)
        y = np.array([self.exp_score(a, self.diffs[item], self.thresholds)
                      for a in abilities]).reshape(-1, 1)

        return self.plot_data(
            x_data=abilities, y_data=y, x_obs_data=xobsdata, y_obs_data=yobsdata,
            x_min=xmin, x_max=xmax, y_max=self.max_score, items=item,
            graph_title=title or '', y_label='Expected score', obs=obs,
            thresh_lines=thresh_lines, central_diff=central_diff,
            score_lines_item=[item, score_lines], score_labels=score_labels,
            plot_style=plot_style, palette=palette, black=black, font=font,
            cat_highlight=cat_highlight, title_font_size=title_font_size,
            axis_font_size=axis_font_size, labelsize=labelsize,
            filename=filename, plot_density=dpi, file_format=file_format
        )

    def crcs(self, item=None, obs=None, no_of_classes=5, title=None,
             thresh_lines=False, central_diff=False, cat_highlight=None,
             xmin=-5, xmax=5, plot_style='white', palette='dark blue',
             black=False, font='Times New Roman', title_font_size=15,
             axis_font_size=12, labelsize=12, filename=None,
             file_format='png', dpi=300):
        """
        Plot Category Response Curves (CRCs) for a single item.

        Displays the probability of each response category as a function of
        ability using the RSM centred parameterisation. Optionally overlays
        observed category proportions.

        Parameters
        ----------
        item : str or None, default None
            Item identifier. If None, uses zero difficulty.
        obs : list, 'all', or None, default None
            Observed overlay: 'all', list of category indices, or None.
        no_of_classes : int, default 5
            Number of class intervals.
        title : str or None, default None
            Plot title.
        thresh_lines : bool, default False
            Draw vertical lines at absolute threshold locations.
        central_diff : bool, default False
            Draw a line at the item central difficulty.
        cat_highlight : int or None, default None
            Category to shade.
        xmin, xmax : float
            Ability axis limits.
        plot_style, palette, black, font : see plot_data().
        title_font_size, axis_font_size, labelsize : int
            Font sizes.
        filename : str or None
            If provided, saves the plot.
        file_format : str, default 'png'
            Output format.
        dpi : int, default 300
            Resolution.

        Returns
        -------
        matplotlib.figure.Figure
        """
        if item == 'none':
            item = None
        # BUG FIX: typo 'person_abiliites'
        if obs is not None and not hasattr(self, 'person_abilities'):
            self.person_abils(warm_corr=False)

        xobsdata = yobsdata = np.array(np.nan)
        if obs is not None:
            xobsdata, yobsdata = self.class_intervals_cats(
                self.person_abilities, item=item, no_of_classes=no_of_classes
            )
            if isinstance(obs, str) and obs == 'all':
                obs = np.arange(self.max_score + 1)
            if not all(c in np.arange(self.max_score + 1) for c in obs):
                warnings.warn("Invalid 'obs' value. Valid values are None, 'all', "
                              'or a list of category indices.',
                              UserWarning, stacklevel=2)
                return
            yobsdata = yobsdata[:, obs]

        abilities = np.arange(-20, 20, 0.1)
        diff = 0 if item is None else self.diffs[item]
        y = np.array([[self.cat_prob(a, diff, cat, self.thresholds)
                       for cat in range(self.max_score + 1)]
                      for a in abilities])

        return self.plot_data(
            x_data=abilities, y_data=y, x_min=xmin, x_max=xmax, y_max=1,
            x_obs_data=xobsdata, y_obs_data=yobsdata, items=item,
            graph_title=title or '', y_label='Probability', obs=obs,
            thresh_lines=thresh_lines, central_diff=central_diff,
            cat_highlight=cat_highlight, plot_style=plot_style,
            palette=palette, black=black, font=font,
            title_font_size=title_font_size, axis_font_size=axis_font_size,
            labelsize=labelsize, filename=filename, plot_density=dpi,
            file_format=file_format
        )

    def threshold_ccs(self, item=None, obs=None, no_of_classes=5, title=None,
                      thresh_lines=False, central_diff=False, cat_highlight=None,
                      xmin=-5, xmax=5, plot_style='white', palette='dark blue',
                      black=False, font='Times New Roman', title_font_size=15,
                      axis_font_size=12, labelsize=12, filename=None,
                      file_format='png', dpi=300):
        """
        Plot Threshold Characteristic Curves (TCCs).

        Displays the probability of scoring in the higher of two adjacent
        categories at each shared threshold. When item=None, plots thresholds
        at their shared locations without item difficulty offset.

        Parameters
        ----------
        item : str or None, default None
            Item identifier. If None, plots at shared threshold locations.
        obs : list, 'all', or None, default None
            Observed overlay: 'all', list of 1-based threshold numbers, or None.
        no_of_classes : int, default 5
            Number of class intervals.
        title : str or None, default None
            Plot title.
        thresh_lines : bool, default False
            Draw vertical lines at threshold locations.
        central_diff : bool, default False
            Draw a line at the item central difficulty.
        cat_highlight : int or None, default None
            Threshold category to shade.
        xmin, xmax : float
            Ability axis limits.
        plot_style, palette, black, font : see plot_data().
        title_font_size, axis_font_size, labelsize : int
            Font sizes.
        filename : str or None
            If provided, saves the plot.
        file_format : str, default 'png'
            Output format.
        dpi : int, default 300
            Resolution.

        Returns
        -------
        matplotlib.figure.Figure
        """
        if item == 'none':
            item = None
        # BUG FIX: typo 'person_abiliites'
        if obs is not None and not hasattr(self, 'person_abilities'):
            self.person_abils(warm_corr=False)

        xobsdata = yobsdata = np.array(np.nan)
        if obs is not None:
            mean_abilities, obs_props = self.class_intervals_thresholds(
                item=item, no_of_classes=no_of_classes
            )
            xobsdata, yobsdata = mean_abilities, obs_props
            if obs != 'all':
                if not all(c in np.arange(self.max_score) + 1 for c in obs):
                    warnings.warn("Invalid 'obs' value. Valid values are None, 'all', "
                                  'or a list of threshold numbers.',
                                  UserWarning, stacklevel=2)
                    return
                obs_idx  = [o - 1 for o in obs]
                xobsdata = xobsdata[:, obs_idx]
                yobsdata = yobsdata[:, obs_idx]

        abilities = np.arange(-20, 20, 0.1)
        # Absolute threshold locations: tau_k (+ item difficulty if item-specific)
        abs_thresh = (self.thresholds[1:] if item is None
                      else self.thresholds[1:] + self.diffs[item])
        y = np.array([[1.0 / (1.0 + np.exp(thr - a)) for thr in abs_thresh]
                      for a in abilities])

        return self.plot_data(
            x_data=abilities, y_data=y, y_max=1, x_min=xmin, x_max=xmax,
            items=item, obs=obs, x_obs_data=xobsdata, y_obs_data=yobsdata,
            graph_title=title or '', y_label='Probability',
            thresh_lines=thresh_lines, central_diff=central_diff,
            cat_highlight=cat_highlight, plot_style=plot_style,
            palette=palette, black=black, font=font,
            title_font_size=title_font_size, axis_font_size=axis_font_size,
            labelsize=labelsize, filename=filename, file_format=file_format,
            plot_density=dpi
        )

    def iic(self, item, ymax=None, thresh_lines=False, central_diff=False,
            point_info_lines=None, point_info_labels=False, cat_highlight=None,
            title=None, xmin=-5, xmax=5, plot_style='white', palette='dark blue',
            black=False, font='Times New Roman', title_font_size=15,
            axis_font_size=12, labelsize=12, filename=None,
            file_format='png', dpi=300):
        """
        Plot the Item Information Curve (IIC) for a single item.

        Displays Fisher information as a function of ability.

        Parameters
        ----------
        item : str
            Item identifier.
        ymax : float or None, default None
            Upper y-axis limit. Auto-scaled if None.
        thresh_lines : bool, default False
            Draw vertical lines at absolute threshold locations.
        central_diff : bool, default False
            Draw a line at the item central difficulty.
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
        filename : str or None
            If provided, saves the plot.
        file_format : str, default 'png'
            Output format.
        dpi : int, default 300
            Resolution.

        Returns
        -------
        matplotlib.figure.Figure
        """
        abilities = np.arange(-20, 20, 0.1)
        y = np.array([self.variance(a, self.diffs[item], self.thresholds)
                      for a in abilities]).reshape(-1, 1)
        if ymax is None:
            ymax = float(y.max()) * 1.1

        return self.plot_data(
            x_data=abilities, y_data=y, x_min=xmin, x_max=xmax, y_max=ymax,
            items=item, thresh_lines=thresh_lines, central_diff=central_diff,
            point_info_lines_item=[item, point_info_lines],
            score_labels=point_info_labels, cat_highlight=cat_highlight,
            graph_title=title or '', y_label='Fisher information',
            plot_style=plot_style, palette=palette, black=black, font=font,
            title_font_size=title_font_size, axis_font_size=axis_font_size,
            labelsize=labelsize, filename=filename, plot_density=dpi,
            file_format=file_format
        )

    def tcc(self, items=None, obs=False, no_of_classes=5, title=None,
            score_lines=None, score_labels=False, xmin=-5, xmax=5,
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
        no_of_classes : int, default 5
            Number of class intervals.
        title : str or None, default None
            Plot title.
        score_lines : list or None, default None
            Raw total scores at which to draw reference lines.
        score_labels : bool, default False
            Annotate score line intersections.
        xmin, xmax : float
            Ability axis limits.
        plot_style, palette, black, font : see plot_data().
        title_font_size, axis_font_size, labelsize : int
            Font sizes.
        filename : str or None
            If provided, saves the plot.
        file_format : str, default 'png'
            Output format.
        dpi : int, default 300
            Resolution.

        Returns
        -------
        matplotlib.figure.Figure
        """
        if isinstance(items, str) and items in ('all', 'none'):
            items = None
        elif isinstance(items, str):
            items = [items]

        # BUG FIX: typo 'person_abiliites'
        if obs and not hasattr(self, 'person_abilities'):
            self.person_abils(warm_corr=False)

        xobsdata = yobsdata = np.array(np.nan)
        if obs:
            mean_abilities, obs_means = self.class_intervals(
                items=items, no_of_classes=no_of_classes
            )
            xobsdata = mean_abilities
            yobsdata = np.array(obs_means).reshape(no_of_classes, 1)

        abilities  = np.arange(-20, 20, 0.1)
        item_keys  = (list(self.dataframe.columns) if items is None else items)
        y = np.array([sum(self.exp_score(a, self.diffs[it], self.thresholds)
                          for it in item_keys)
                      for a in abilities]).reshape(-1, 1)
        y_max = self.max_score * len(item_keys)

        return self.plot_data(
            x_data=abilities, y_data=y, items=items,
            x_obs_data=xobsdata, y_obs_data=yobsdata,
            x_min=xmin, x_max=xmax, y_max=y_max,
            score_lines_test=score_lines, score_labels=score_labels,
            graph_title=title or '', y_label='Expected score', obs=obs,
            plot_style=plot_style, palette=palette, black=black, font=font,
            title_font_size=title_font_size, axis_font_size=axis_font_size,
            labelsize=labelsize, filename=filename, plot_density=dpi,
            file_format=file_format
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
        filename : str or None
            If provided, saves the plot.
        file_format : str, default 'png'
            Output format.
        dpi : int, default 300
            Resolution.

        Returns
        -------
        matplotlib.figure.Figure
        """
        if isinstance(items, str) and items in ('all', 'none'):
            items = None
        elif isinstance(items, str):
            items = [items]
        item_keys = (list(self.dataframe.columns) if items is None else items)
        abilities = np.arange(-20, 20, 0.1)
        y = np.array([sum(self.variance(a, self.diffs[it], self.thresholds)
                          for it in item_keys)
                      for a in abilities]).reshape(-1, 1)
        if ymax is None:
            ymax = float(y.max()) * 1.1

        return self.plot_data(
            x_data=abilities, y_data=y, items=items,
            x_min=xmin, x_max=xmax, y_max=ymax,
            graph_title=title or '', point_info_lines_test=point_info_lines,
            score_labels=point_info_labels, y_label='Fisher information',
            plot_style=plot_style, palette=palette, black=black, font=font,
            title_font_size=title_font_size, axis_font_size=axis_font_size,
            labelsize=labelsize, filename=filename, plot_density=dpi,
            file_format=file_format
        )

    def test_csem(self, items=None, point_csem_lines=None, point_csem_labels=False,
                  xmin=-5, xmax=5, ymax=5, title=None, plot_style='white',
                  palette='dark blue', black=False, font='Times New Roman',
                  title_font_size=15, axis_font_size=12, labelsize=12,
                  filename=None, file_format='png', dpi=300):
        """
        Plot the Test Conditional Standard Error of Measurement (CSEM) Curve.

        Displays 1 / sqrt(I(theta)) as a function of ability.

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
        filename : str or None
            If provided, saves the plot.
        file_format : str, default 'png'
            Output format.
        dpi : int, default 300
            Resolution.

        Returns
        -------
        matplotlib.figure.Figure
        """
        if isinstance(items, str) and items in ('all', 'none'):
            items = None
        elif isinstance(items, str):
            items = [items]
        item_keys = (list(self.dataframe.columns) if items is None else items)
        abilities = np.arange(-20, 20, 0.1)
        info = np.array([sum(self.variance(a, self.diffs[it], self.thresholds)
                             for it in item_keys)
                         for a in abilities])
        y = (1.0 / (info ** 0.5)).reshape(-1, 1)

        return self.plot_data(
            x_data=abilities, y_data=y, items=items,
            x_min=xmin, x_max=xmax, y_max=ymax,
            graph_title=title or '', point_csem_lines=point_csem_lines,
            score_labels=point_csem_labels, y_label='Conditional SEM',
            plot_style=plot_style, palette=palette, black=black, font=font,
            title_font_size=title_font_size, axis_font_size=axis_font_size,
            labelsize=labelsize, filename=filename, plot_density=dpi,
            file_format=file_format
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

        Displays the distribution of standardised residuals. Under a
        well-fitting Rasch model these approximate a standard normal.
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
        filename : str or None
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