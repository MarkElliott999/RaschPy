from math import exp, log, sqrt, floor
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


class MFRM(Rasch):
    '''
    Many-Facet Rasch Model (Linacre 1994) with RSM (Andrich 1978) formulation.

    Supports four rater severity parameterisations:
      'global'     — scalar severity σ_r per rater
      'items'      — vector σ_{r,i} per (rater, item)
      'thresholds' — vector σ_{r,k} per (rater, threshold)
      'matrix'     — full σ_{r,i,k} per (rater, item, threshold)

    The log-numerator for person n, rater r, item i, category k is:
      global:     k*(θ_n − δ_i − σ_r) − Σ τ_k
      items:      k*(θ_n − δ_i − σ_{r,i}) − Σ τ_k
      thresholds: k*(θ_n − δ_i) − Σ(τ_k + σ_{r,k})
      matrix:     k*(θ_n − δ_i) − Σ(τ_k + σ_{r,i,k})

    Data format: (Rater, Person) MultiIndex × Items DataFrame.
    '''

    # ------------------------------------------------------------------
    # Model registry — maps model name to severity attribute names
    # ------------------------------------------------------------------
    _MODELS = ('global', 'items', 'thresholds', 'matrix')

    def _attr(self, model, name, anchor=False):
        '''Return the attribute name for a given model and statistic.'''
        prefix = 'anchor_' if anchor else ''
        suffix = f'_{model}'
        return f'{prefix}{name}{suffix}'

    def _get_params(self, model, anchor=False):
        '''
        Return (difficulties, thresholds, severities) for the requested model.
        Auto-triggers calibration if not yet run.
        '''
        if anchor:
            diff_attr  = f'anchor_diffs_{model}'
            thr_attr   = f'anchor_thresholds_{model}'
            sev_attr   = f'anchor_severities_{model}'
            if not hasattr(self, diff_attr):
                raise AttributeError(
                    f'Anchor calibration required. '
                    f'Run self.calibrate_{model}_anchor().'
                )
        else:
            diff_attr = 'diffs'
            thr_attr  = 'thresholds'
            sev_attr  = f'severities_{model}'
            if not hasattr(self, sev_attr):
                self.calibrate(model=model)
        return (getattr(self, diff_attr),
                getattr(self, thr_attr),
                getattr(self, sev_attr))

    def _get_abils(self, model, anchor=False):
        '''Return ability estimates for the requested model. Auto-triggers if needed.'''
        attr = f'anchor_abils_{model}' if anchor else f'abils_{model}'
        if not hasattr(self, attr):
            self.person_abils(model=model, anchor=anchor)
        return getattr(self, attr)

    
    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def __init__(self,
                 dataframe,
                 max_score=0,
                 extreme_persons=True,
                 no_of_classes=5):
        """
        Initialise a Many-Facet Rasch Model (MFRM) object.

        The MFRM extends the RSM/PCM to include rater facets. Four rater
        parameterisations are supported, selected at calibrate() time:
        'global' (single severity per rater), 'items' (per-item severities),
        'thresholds' (per-threshold severities), 'matrix' (per-item-threshold).

        Parameters
        ----------
        dataframe : pandas.DataFrame
            Response data with a (Rater, Person) MultiIndex and items as
            columns. Cell values should be non-negative integers from 0 to
            max_score; NaN for missing responses.
        max_score : int, default 0
            Maximum possible score per item. 0 means auto-detect from the
            data (np.nanmax). Supply explicitly to avoid issues when the
            maximum is never observed.
        extreme_persons : bool, default True
            If True, removes only persons with entirely missing data across
            all raters. If False, additionally removes persons with all-zero
            or perfect total scores.
        no_of_classes : int, default 5
            Number of class intervals for observed-data overlays on plots.

        Attributes set
        --------------
        dataframe : pandas.DataFrame
            Filtered response data with (Rater, Person) MultiIndex.
        invalid_responses : pandas.DataFrame
            Rows removed based on the extreme_persons rule.
        max_score : int
            Maximum possible score per item.
        no_of_persons : int
            Number of unique persons after filtering.
        no_of_items : int
            Number of items (columns).
        no_of_raters : int
            Number of unique raters.
        no_of_classes : int
            Number of class intervals for plots.
        items : pandas.Index
            Item identifiers (column names).
        raters : pandas.Index
            Rater identifiers.
        persons : pandas.Index
            Person identifiers.
        anchor_raters_{model} : list
            Empty list per model (global/items/thresholds/matrix) for
            anchor rater tracking.
        """
        self.max_score = int(np.nanmax(dataframe)) if max_score == 0 else max_score

        unstacked_df = dataframe.unstack(level=0)

        if extreme_persons:
            to_drop = unstacked_df[unstacked_df.isna().all(axis=1)].index
        else:
            scores     = unstacked_df.sum(axis=1)
            max_scores = unstacked_df.notna().sum(axis=1) * self.max_score
            to_drop    = unstacked_df[(scores == 0) | (scores == max_scores)].index

        self.invalid_responses = dataframe[
            dataframe.index.get_level_values(1).isin(to_drop)
        ]
        self.dataframe = dataframe[
            ~dataframe.index.get_level_values(1).isin(to_drop)
        ]

        self.no_of_persons = len(self.dataframe.index.levels[1])
        self.no_of_items   = self.dataframe.shape[1]
        self.no_of_raters  = len(self.dataframe.index.levels[0])
        self.no_of_classes = no_of_classes
        self.items   = self.dataframe.columns
        self.raters  = self.dataframe.index.get_level_values(0).unique()
        self.persons = self.dataframe.index.get_level_values(1).unique()

        # Anchor rater tracking per model
        for model in self._MODELS:
            setattr(self, f'anchor_raters_{model}', [])

    # ------------------------------------------------------------------
    # Rename utilities
    # ------------------------------------------------------------------

    def rename_rater(self, old, new):
        """
        Rename a single rater in the dataframe.

        Validates the rename (no duplicates, no self-rename, must be a string)
        and updates self.raters. Prints a message rather than raising if
        validation fails.

        Parameters
        ----------
        old : str
            Current rater name.
        new : str
            Desired new rater name.
        """

        if old == new:
            warnings.warn('New rater name is the same as the old rater name.',
                          UserWarning, stacklevel=2)
        elif new in self.raters:
            warnings.warn('New rater name is a duplicate of an existing rater name.',
                          UserWarning, stacklevel=2)
        if old not in self.raters:
            warnings.warn(f'Old rater name {old!r} not found in data.',
                          UserWarning, stacklevel=2)
        elif not isinstance(new, str):
            warnings.warn('Rater names must be strings.',
                          UserWarning, stacklevel=2)
        else:
            new_names = [new if r == old else r for r in self.raters]
            self.rename_raters_all(new_names)

    def rename_raters_all(self, new_names):
        """
        Rename all raters at once.

        Validates the new name list (correct length, no duplicates, all strings)
        and rebuilds the dataframe with the new rater index labels.

        Parameters
        ----------
        new_names : list of str
            New rater names in the same order as self.raters.
        """

        if len(new_names) != len(set(new_names)):
            warnings.warn('List of new rater names contains duplicates.',
                          UserWarning, stacklevel=2)
        elif len(new_names) != self.no_of_raters:
            warnings.warn(f'Incorrect number of rater names: {len(new_names)} provided, '
                          f'{self.no_of_raters} raters in data.',
                          UserWarning, stacklevel=2)
        elif not all(isinstance(n, str) for n in new_names):
            warnings.warn('Rater names must be strings.',
                          UserWarning, stacklevel=2)
        else:
            df_dict = {new: self.dataframe.xs(old)
                       for old, new in zip(self.raters, new_names)}
            self.dataframe = pd.concat(df_dict.values(), keys=df_dict.keys())
            self.raters = self.dataframe.index.get_level_values(0).unique()

    def rename_person(self, old, new):
        """
        Rename a single person in the dataframe.

        Validates the rename and updates the level-1 (Person) index.
        Prints a message rather than raising if validation fails.

        Parameters
        ----------
        old : str
            Current person name.
        new : str
            Desired new person name.
        """

        if old == new:
            warnings.warn('New person name is the same as the old person name.',
                          UserWarning, stacklevel=2)
        elif new in self.persons:
            warnings.warn('New person name is a duplicate of an existing person name.',
                          UserWarning, stacklevel=2)
        if old not in self.persons:
            warnings.warn(f'Old person name {old!r} not found in data.',
                          UserWarning, stacklevel=2)
        elif not isinstance(new, str):
            warnings.warn('Person names must be strings.',
                          UserWarning, stacklevel=2)
        else:
            self.dataframe = self.dataframe.rename(
                index={old: new}, level=1
            )
            self.persons = self.dataframe.index.get_level_values(1).unique()

    def rename_persons_all(self, new_names):
        """
        Rename all persons at once.

        Validates the new name list and rebuilds the level-1 (Person) index.

        Parameters
        ----------
        new_names : list of str
            New person names in the same order as self.persons.
        """

        if len(new_names) != len(set(new_names)):
            warnings.warn('List of new person names contains duplicates.',
                          UserWarning, stacklevel=2)
        elif len(new_names) != self.no_of_persons:
            warnings.warn(f'Incorrect number of person names: {len(new_names)} provided, '
                          f'{self.no_of_persons} persons in data.',
                          UserWarning, stacklevel=2)
        elif not all(isinstance(n, str) for n in new_names):
            warnings.warn('Person names must be strings.',
                          UserWarning, stacklevel=2)
        else:
            rename_map = dict(zip(self.persons, new_names))
            self.dataframe = self.dataframe.rename(index=rename_map, level=1)
            self.persons = self.dataframe.index.get_level_values(1).unique()

    # ------------------------------------------------------------------
    # Scalar probability functions (used in plots)
    # ------------------------------------------------------------------

    def cat_prob(self, ability, item, difficulties, rater, severities,
                 category, thresholds, model='global'):
        """
        Compute the probability of a response category for a single observation.

        Applies the MFRM log-numerator: k*(b - d_i) - cumsum(tau) - rater_severity,
        where the severity term depends on the model parameterisation.
        Numerically stabilised via log-sum-exp.

        Parameters
        ----------
        ability : float
            Person ability estimate on the logit scale.
        item : str
            Item identifier.
        difficulties : pandas.Series
            Item difficulty estimates indexed by item name.
        rater : str
            Rater identifier.
        severities : Series or dict
            Rater severity parameters. Structure depends on model:
            global — Series indexed by rater;
            items  — dict of Series {rater: Series(items)};
            thresholds — dict of arrays {rater: array(thresholds)};
            matrix — nested dict {rater: {item: array}}.
        category : int
            Response category (0 to max_score).
        thresholds : array-like
            Rasch-Andrich threshold vector, length max_score+1, thresholds[0]=0.
        model : str, default 'global'
            Rater parameterisation: 'global', 'items', 'thresholds', or 'matrix'.

        Returns
        -------
        float
            Probability of the specified category, in [0, 1].
        """
        cats   = np.arange(len(thresholds), dtype=float)
        cumtau = np.cumsum(thresholds)
        log_nums = cats * (ability - difficulties.loc[item]) - cumtau
        # Apply rater severity
        if model == 'global':
            log_nums -= cats * severities.loc[rater]
        elif model == 'items':
            log_nums -= cats * severities[rater][item]
        elif model == 'thresholds':
            log_nums -= np.cumsum(severities[rater])
        elif model == 'matrix':
            log_nums -= np.cumsum(severities[rater][item])
        log_nums -= log_nums.max()
        nums = np.exp(log_nums)
        return nums[category] / nums.sum()

    def exp_score(self, ability, item, difficulties, rater, severities,
                  thresholds, model='global'):
        """
        Compute the expected score for a single person/rater/item combination.

        Calculates E[X | ability, item, rater, model] = sum(k * P(X=k)).
        Used in scalar Newton-Raphson estimation and score_abil().

        Parameters
        ----------
        ability : float
            Person ability estimate on the logit scale.
        item : str
            Item identifier.
        difficulties : pandas.Series
            Item difficulty estimates.
        rater : str
            Rater identifier.
        severities : Series or dict
            Rater severity parameters (structure depends on model).
        thresholds : array-like
            Rasch-Andrich threshold vector, length max_score+1.
        model : str, default 'global'
            Rater parameterisation.

        Returns
        -------
        float
            Expected score in [0, max_score].
        """
        cats = np.arange(len(thresholds), dtype=float)
        probs = np.array([self.cat_prob(ability, item, difficulties, rater,
                                        severities, cat, thresholds, model)
                          for cat in range(len(thresholds))])
        return (cats * probs).sum()

    def variance(self, ability, item, difficulties, rater, severities,
                 thresholds, model='global'):
        """
        Compute item variance (Fisher information) for a single observation.

        Calculates Var[X | ability, item, rater, model] = sum((k - E[X])^2 * P(X=k)).
        Used in scalar Newton-Raphson estimation and score_abil().

        Parameters
        ----------
        ability : float
            Person ability estimate on the logit scale.
        item : str
            Item identifier.
        difficulties : pandas.Series
            Item difficulty estimates.
        rater : str
            Rater identifier.
        severities : Series or dict
            Rater severity parameters.
        thresholds : array-like
            Rasch-Andrich threshold vector.
        model : str, default 'global'
            Rater parameterisation.

        Returns
        -------
        float
            Item variance / Fisher information. Always non-negative.
        """
        cats  = np.arange(len(thresholds), dtype=float)
        probs = np.array([self.cat_prob(ability, item, difficulties, rater,
                                        severities, cat, thresholds, model)
                          for cat in range(len(thresholds))])
        exp   = (cats * probs).sum()
        return ((cats - exp) ** 2 * probs).sum()

    def kurtosis(self, ability, item, difficulties, rater, severities,
                 thresholds, model='global'):
        """
        Compute the fourth central moment for a single person/rater/item.

        Calculates sum((k - E[X])^4 * P(X=k)). Used in Wilson-Hilferty
        approximation for standardised fit statistics.

        Parameters
        ----------
        ability : float
            Person ability estimate on the logit scale.
        item : str
            Item identifier.
        difficulties : pandas.Series
            Item difficulty estimates.
        rater : str
            Rater identifier.
        severities : Series or dict
            Rater severity parameters.
        thresholds : array-like
            Rasch-Andrich threshold vector.
        model : str, default 'global'
            Rater parameterisation.

        Returns
        -------
        float
            Fourth central moment of the response distribution.
        """
        cats  = np.arange(len(thresholds), dtype=float)
        probs = np.array([self.cat_prob(ability, item, difficulties, rater,
                                        severities, cat, thresholds, model)
                          for cat in range(len(thresholds))])
        exp   = (cats * probs).sum()
        return ((cats - exp) ** 4 * probs).sum()

    # Backwards-compatible aliases for the four parameterisations
    def cat_prob_global(self, a, i, d, r, s, c, t):
        return self.cat_prob(a, i, d, r, s, c, t, 'global')
    def cat_prob_items(self, a, i, d, r, s, c, t):
        return self.cat_prob(a, i, d, r, s, c, t, 'items')
    def cat_prob_thresholds(self, a, i, d, r, s, c, t):
        return self.cat_prob(a, i, d, r, s, c, t, 'thresholds')
    def cat_prob_matrix(self, a, i, d, r, s, c, t):
        return self.cat_prob(a, i, d, r, s, c, t, 'matrix')

    def exp_score_global(self, a, i, d, r, s, t):
        return self.exp_score(a, i, d, r, s, t, 'global')
    def exp_score_items(self, a, i, d, r, s, t):
        return self.exp_score(a, i, d, r, s, t, 'items')
    def exp_score_thresholds(self, a, i, d, r, s, t):
        return self.exp_score(a, i, d, r, s, t, 'thresholds')
    def exp_score_matrix(self, a, i, d, r, s, t):
        return self.exp_score(a, i, d, r, s, t, 'matrix')

    def variance_global(self, a, i, d, r, s, t):
        return self.variance(a, i, d, r, s, t, 'global')
    def variance_items(self, a, i, d, r, s, t):
        return self.variance(a, i, d, r, s, t, 'items')
    def variance_thresholds(self, a, i, d, r, s, t):
        return self.variance(a, i, d, r, s, t, 'thresholds')
    def variance_matrix(self, a, i, d, r, s, t):
        return self.variance(a, i, d, r, s, t, 'matrix')

    def kurtosis_global(self, a, i, d, r, s, t):
        return self.kurtosis(a, i, d, r, s, t, 'global')
    def kurtosis_items(self, a, i, d, r, s, t):
        return self.kurtosis(a, i, d, r, s, t, 'items')
    def kurtosis_thresholds(self, a, i, d, r, s, t):
        return self.kurtosis(a, i, d, r, s, t, 'thresholds')
    def kurtosis_matrix(self, a, i, d, r, s, t):
        return self.kurtosis(a, i, d, r, s, t, 'matrix')

    # ------------------------------------------------------------------
    # Vectorised probability engine
    # ------------------------------------------------------------------

    def _cat_probs_mfrm(self, abilities, items, raters, thresholds,
                        model, severities):
        '''
        Vectorised MFRM category probability engine.

        Returns dict {rater: ndarray (K+1, N, I)} and cats array (K+1,).

        The log-numerator for person n, rater r, item i, category k:
          global:     k*(θ_n − δ_i − σ_r) − Σ τ_k
          items:      k*(θ_n − δ_i − σ_{r,i}) − Σ τ_k
          thresholds: k*(θ_n − δ_i) − Σ(τ_k + σ_{r,k})
          matrix:     k*(θ_n − δ_i) − Σ(τ_k + σ_{r,i,k})
        '''
        cats      = np.arange(len(thresholds), dtype=float)   # (K+1,)
        cumtau    = np.cumsum(thresholds)                       # (K+1,)
        ab        = np.asarray(abilities, dtype=float)          # (N,)
        diff_arr  = self.diffs.loc[items].values                # (I,)
        n_items   = len(items)

        result = {}
        for rater in raters:
            if model == 'global':
                # item_offset: scalar, same for all (i)
                item_offset = float(severities.loc[rater])
                thresh_offset = np.zeros(len(thresholds))
            elif model == 'items':
                # item_offset: (I,) vector
                item_offset = np.array(
                    [severities[rater][item] for item in items], dtype=float
                )
                thresh_offset = np.zeros(len(thresholds))
            elif model == 'thresholds':
                item_offset = 0.0
                thresh_offset = np.asarray(severities[rater], dtype=float)
            elif model == 'matrix':
                item_offset = 0.0
                thresh_offset = None  # applied per-item below
            else:
                raise ValueError(f'Unknown model: {model}')

            if model == 'matrix':
                # Build (K+1, N, I) tensor item by item
                log_num = np.zeros((len(thresholds), len(ab), n_items))
                for j, item in enumerate(items):
                    sev_rik = np.asarray(severities[rater][item], dtype=float)
                    cumtau_total = cumtau + np.cumsum(sev_rik)
                    log_num[:, :, j] = (
                        cats[:, None] * (ab[None, :] - diff_arr[j])
                        - cumtau_total[:, None]
                    )
            else:
                if isinstance(item_offset, np.ndarray):
                    io = item_offset[None, None, :]   # (1, 1, I)
                else:
                    io = float(item_offset)
                cumtau_total = cumtau + thresh_offset  # (K+1,)
                log_num = (
                    cats[:, None, None]
                    * (ab[None, :, None] - diff_arr[None, None, :] - io)
                    - cumtau_total[:, None, None]
                )  # (K+1, N, I)

            log_num -= log_num.max(axis=0, keepdims=True)
            probs    = np.exp(log_num)
            probs   /= probs.sum(axis=0, keepdims=True)
            result[rater] = probs

        return result, cats

    # ------------------------------------------------------------------
    # Calibration — shared components
    # ------------------------------------------------------------------

    def _remove_null_persons(self):
        '''Vectorised null person removal.'''
        _pd = self.dataframe.unstack(level=0)
        _null = _pd.isnull().all(axis=1)
        self.null_persons = _pd.index[_null].tolist()
        if self.null_persons:
            self.dataframe = self.dataframe.drop(self.null_persons, level=1)
            self.persons   = self.dataframe.index.get_level_values(1).unique()
        self.no_of_persons = len(self.persons)

    def item_diffs(self, constant=0.1, method='cos', matrix_power=3,
                   log_lik_tol=0.000001):
        '''PAIR item difficulty estimation summing across raters.'''
        data = (self.dataframe.values
                .reshape(self.no_of_raters, self.no_of_persons, -1)
                .swapaxes(1, 2)
                .transpose((1, 0, 2)))  # (I, R, P)

        matrix = np.array([
            [sum(np.count_nonzero(data[i, r, :] == data[j, r, :] + 1)
                 for r in range(self.no_of_raters))
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

    def _threshold_distance(self, threshold, difficulties, constant=0.1):
        '''
        CPAT threshold distance estimate for MFRM — sums counts across raters.
        Vectorised via indicator matrix multiplication.
        '''
        data = (self.dataframe.values
                .reshape(self.no_of_raters, self.no_of_persons, -1)
                .swapaxes(1, 2)
                .transpose((1, 0, 2)))  # (I, R, P)

        # Sum count matrices across raters
        num_matrix = np.zeros((self.no_of_items, self.no_of_items))
        den_matrix = np.zeros((self.no_of_items, self.no_of_items))
        for r in range(self.no_of_raters):
            at_k   = (data[:, r, :] == threshold).astype(np.float64)
            at_km1 = (data[:, r, :] == threshold - 1).astype(np.float64)
            at_kp1 = (data[:, r, :] == threshold + 1).astype(np.float64)
            num_matrix += at_k @ at_k.T
            den_matrix += at_km1 @ at_kp1.T

        valid = (num_matrix + den_matrix) > 0
        num_s = np.where(valid, num_matrix + constant, 0.0)
        den_s = np.where(valid, den_matrix + constant, 0.0)

        with np.errstate(divide='ignore', invalid='ignore'):
            weight_matrix = np.where(valid,
                                     2.0 * num_s * den_s / (num_s + den_s), 0.0)

        diffs = difficulties.values
        diff_matrix = diffs[:, None] - diffs[None, :]

        with np.errstate(divide='ignore', invalid='ignore'):
            log_ratio = np.where(valid, np.log(num_s) - np.log(den_s), 0.0)

        total_weight = weight_matrix.sum()
        if total_weight == 0:
            return np.nan
        return (weight_matrix * (log_ratio + diff_matrix)).sum() / total_weight

    def ra_thresholds(self, difficulties, constant=0.1):
        '''CPAT threshold set estimation.'''
        distances = [self._threshold_distance(k, difficulties, constant)
                     for k in range(1, self.max_score)]
        thresholds = np.array([sum(distances[:t]) for t in range(self.max_score)])
        thresholds -= thresholds.mean()
        return np.insert(thresholds, 0, 0.0)

    # ------------------------------------------------------------------
    # Rater severity estimation
    # ------------------------------------------------------------------

    def _pair_matrix(self, data_2d, constant):
        '''Build a PAIR pairwise matrix from (R, P) data and apply smoothing.'''
        R = data_2d.shape[0]
        matrix = np.array([
            [np.count_nonzero(data_2d[r1, :] == data_2d[r2, :] + 1)
             for r2 in range(R)]
            for r1 in range(R)
        ], dtype=np.float64)
        constant_matrix = ((matrix + matrix.T) > 0).astype(np.float64) * constant
        matrix += constant_matrix
        np.fill_diagonal(matrix, matrix.diagonal() + constant)
        return matrix

    def _raise_matrix_power(self, matrix, matrix_power, constant):
        mat = np.linalg.matrix_power(matrix, matrix_power)
        mat_pow = matrix_power
        while 0 in mat:
            mat = mat @ matrix
            mat_pow += 1
            if mat_pow == matrix_power + 5:
                mat += constant
                break
        return mat

    def raters_global(self, constant=0.1, method='cos', matrix_power=3,
                      log_lik_tol=0.000001):
        '''PAIR rater severity estimation — scalar per rater.'''
        data = (self.dataframe.values
                .reshape(self.no_of_raters, self.no_of_persons, -1)
                .swapaxes(1, 2)
                .transpose((1, 0, 2)))  # (I, R, P)

        matrix = np.array([
            [sum(np.count_nonzero(data[item, r1, :] == data[item, r2, :] + 1)
                 for item in range(self.no_of_items))
             for r2 in range(self.no_of_raters)]
            for r1 in range(self.no_of_raters)
        ], dtype=np.float64)
        constant_matrix = ((matrix + matrix.T) > 0).astype(np.float64) * constant
        matrix += constant_matrix
        np.fill_diagonal(matrix, matrix.diagonal() + constant)

        mat = self._raise_matrix_power(matrix, matrix_power, constant)
        self.severities_global = self.priority_vector(mat, method=method,
                                                      log_lik_tol=log_lik_tol,
                                                      raters=True)

    def _item_rater_element(self, item, constant=0.1, method='cos',
                            matrix_power=3, log_lik_tol=0.000001):
        '''PAIR rater severity for a single item (items parameterisation).'''
        data = (self.dataframe.values
                .reshape(self.no_of_raters, self.no_of_persons, -1)
                .swapaxes(1, 2)
                .transpose((1, 0, 2)))  # (I, R, P)
        matrix = self._pair_matrix(data[item, :, :], constant)
        mat    = self._raise_matrix_power(matrix, matrix_power, constant)
        return self.priority_vector(mat, method=method, log_lik_tol=log_lik_tol,
                                    raters=True)

    def raters_items(self, constant=0.1, method='cos', matrix_power=3,
                     log_lik_tol=0.000001):
        '''PAIR rater severity estimation — vector per (rater, item).'''
        raters = np.zeros((self.no_of_raters, self.no_of_items))
        for i in range(self.no_of_items):
            raters[:, i] = self._item_rater_element(
                i, constant=constant, method=method,
                matrix_power=matrix_power, log_lik_tol=log_lik_tol
            )
        raters_df = pd.DataFrame(raters, index=self.raters,
                                 columns=self.dataframe.columns)
        self.severities_items = raters_df.T.to_dict()

    def _threshold_rater_element(self, category, constant=0.1, method='cos',
                                 matrix_power=3, log_lik_tol=0.000001):
        '''PAIR rater severity for a single threshold (thresholds parameterisation).'''
        data = (self.dataframe.values
                .reshape(self.no_of_raters, self.no_of_persons, -1)
                .swapaxes(1, 2)
                .transpose((1, 0, 2)))  # (I, R, P)

        # Sum across items: count(X_{i,r1}==k+1 AND X_{i,r2}==k)
        matrix = np.zeros((self.no_of_raters, self.no_of_raters))
        for i in range(self.no_of_items):
            at_k   = (data[i, :, :] == category + 1).astype(np.float64)  # (R, P)
            at_km1 = (data[i, :, :] == category).astype(np.float64)
            matrix += at_k @ at_km1.T

        matrix = matrix.astype(np.float64)
        constant_matrix = ((matrix + matrix.T) > 0).astype(np.float64) * constant
        matrix += constant_matrix
        np.fill_diagonal(matrix, matrix.diagonal() + constant)

        mat = self._raise_matrix_power(matrix, matrix_power, constant)
        return self.priority_vector(mat, method=method, log_lik_tol=log_lik_tol,
                                    raters=True)

    def raters_thresholds(self, constant=0.1, method='cos', matrix_power=3,
                          log_lik_tol=0.000001):
        '''PAIR rater severity estimation — vector per (rater, threshold).'''
        raters = np.zeros((self.no_of_raters, self.max_score))
        for k in range(self.max_score):
            raters[:, k] = self._threshold_rater_element(
                k, constant=constant, method=method,
                matrix_power=matrix_power, log_lik_tol=log_lik_tol
            )
        raters = np.insert(raters, 0, 0.0, axis=1)
        self.severities_thresholds = {
            rater: sev for rater, sev in zip(self.raters, raters)
        }

    def _matrix_rater_element(self, item, category, constant=0.1, method='cos',
                               matrix_power=3, log_lik_tol=0.000001):
        '''PAIR rater severity for a single (item, category) cell (matrix param).'''
        data = (self.dataframe.values
                .reshape(self.no_of_raters, self.no_of_persons, -1)
                .swapaxes(1, 2)
                .transpose((1, 0, 2)))  # (I, R, P)

        at_k   = (data[item, :, :] == category + 1).astype(np.float64)  # (R, P)
        at_km1 = (data[item, :, :] == category).astype(np.float64)
        matrix = at_k @ at_km1.T

        matrix = matrix.astype(np.float64)
        constant_matrix = ((matrix + matrix.T) > 0).astype(np.float64) * constant
        matrix += constant_matrix
        np.fill_diagonal(matrix, matrix.diagonal() + constant)

        mat = self._raise_matrix_power(matrix, matrix_power, constant)
        return self.priority_vector(mat, method=method, log_lik_tol=log_lik_tol,
                                    raters=True)

    def raters_matrix(self, constant=0.1, method='cos', matrix_power=3,
                      log_lik_tol=0.000001):
        '''PAIR rater severity estimation — full (rater, item, threshold) matrix.'''
        raters = np.zeros((self.no_of_raters, self.no_of_items, self.max_score + 1))
        for i in range(self.no_of_items):
            for k in range(self.max_score):
                raters[:, i, k + 1] = self._matrix_rater_element(
                    i, k, constant=constant, method=method,
                    matrix_power=matrix_power, log_lik_tol=log_lik_tol
                )

        rater_dict = {
            rater: {
                item: raters[r, j, :]
                for j, item in enumerate(self.dataframe.columns)
            }
            for r, rater in enumerate(self.raters)
        }

        # Marginal severities for use in plots and stats
        sev_arr = raters[:, :, 1:]  # (R, I, K)
        self.marginal_severities_items = {
            rater: pd.Series({
                item: sev_arr[r, j, :].mean()
                for j, item in enumerate(self.dataframe.columns)
            })
            for r, rater in enumerate(self.raters)
        }
        self.marginal_severities_thresholds = {
            rater: pd.Series(
                np.concatenate([[0.0], sev_arr[r, :, :].mean(axis=0)])
            )
            for r, rater in enumerate(self.raters)
        }
        self.severities_matrix = rater_dict

    # ------------------------------------------------------------------
    # Calibration — top-level methods
    # ------------------------------------------------------------------

    def calibrate(self,
                  model='global',
                  constant=0.1,
                  method='cos',
                  matrix_power=3,
                  log_lik_tol=0.000001):
        '''
        Calibrate the MFRM for the specified rater parameterisation.

        Three-stage sequential estimation:
          1. item_diffs()       — PAIR item difficulties (shared across models)
          2. ra_thresholds()    — CPAT shared thresholds (shared across models)
          3. raters_{model}()   — PAIR rater severities (model-specific)

        Parameters
        ----------
        model : one of 'global', 'items', 'thresholds', 'matrix'
        '''
        if model not in self._MODELS:
            raise ValueError(f'model must be one of {self._MODELS}')

        if constant == 0:
            all_max_items = [item for item in self.items
                             if (self.dataframe.xs(item, level=-1, axis=1).dropna(how='all').eq(self.max_score).all(axis=None))]
            if all_max_items:
                warnings.warn(f"Items with all-maximum scores detected with constant=0: "
                              f"{all_max_items}. Item estimation will fail. "
                              f"Either drop these items or use a non-zero constant.",
                              UserWarning, stacklevel=2)

        if len(self.raters) == 1:
            warnings.warn("Only one rater detected. MFRM with a single rater reduces to RSM. "
                          "Consider using RSM instead.",
                          UserWarning, stacklevel=2)

        if len(self.items) == 1:
            warnings.warn("Only one item detected. MFRM with a single item reduces to RSM "
                          "with raters as items. Consider reconfiguring and using RSM instead.",
                          UserWarning, stacklevel=2)

        self._remove_null_persons()
        self.item_diffs(constant=constant, method=method,
                        matrix_power=matrix_power, log_lik_tol=log_lik_tol)
        self.thresholds = self.ra_thresholds(self.diffs, constant=constant)
        getattr(self, f'raters_{model}')(
            constant=constant, method=method,
            matrix_power=matrix_power, log_lik_tol=log_lik_tol
        )

    # Backwards-compatible aliases
    def calibrate_global(self, **kw):
        self.calibrate(model='global', **kw)
    def calibrate_items(self, **kw):
        self.calibrate(model='items', **kw)
    def calibrate_thresholds(self, **kw):
        self.calibrate(model='thresholds', **kw)
    def calibrate_matrix(self, **kw):
        self.calibrate(model='matrix', **kw)

    # ------------------------------------------------------------------
    # Anchor calibration
    # ------------------------------------------------------------------

    def calibrate_anchor(self, model, anchor_raters, calibrate=False,
                         constant=0.1, method='cos', matrix_power=3,
                         log_lik_tol=0.000001):
        '''
        Anchor calibration: set mean severity of anchor_raters to zero
        and adjust item difficulties and thresholds accordingly.
        '''
        if calibrate:
            self.calibrate(model=model, constant=constant, method=method,
                           matrix_power=matrix_power, log_lik_tol=log_lik_tol)

        if model == 'global':
            self._calibrate_anchor_global(anchor_raters)
        elif model == 'items':
            self._calibrate_anchor_items(anchor_raters)
        elif model == 'thresholds':
            self._calibrate_anchor_thresholds(anchor_raters)
        elif model == 'matrix':
            self._calibrate_anchor_matrix(anchor_raters)

        setattr(self, f'anchor_raters_{model}', anchor_raters)

    def _calibrate_anchor_global(self, anchor_raters):
        self.anchor_diffs_global      = self.diffs.copy()
        self.anchor_thresholds_global = self.thresholds.copy()
        self.anchor_severities_global = self.severities_global.copy()

        adj = float(self.severities_global.loc[anchor_raters].mean())
        self.anchor_severities_global -= adj

    def _calibrate_anchor_items(self, anchor_raters):
        self.anchor_diffs_items      = self.diffs.copy()
        self.anchor_thresholds_items = self.thresholds.copy()

        sev_df = pd.DataFrame(self.severities_items).T  # (R, I)
        adj    = sev_df.loc[anchor_raters].mean(axis=0)

        self.anchor_diffs_items += adj
        sev_df -= adj
        self.anchor_severities_items = {
            rater: {item: sev_df.loc[rater, item]
                    for item in self.dataframe.columns}
            for rater in self.raters
        }
        self.anchor_diffs_items -= self.anchor_diffs_items.mean()

    def _calibrate_anchor_thresholds(self, anchor_raters):
        self.anchor_diffs_thresholds      = self.diffs.copy()
        self.anchor_thresholds_thresholds = self.thresholds.copy()

        sev_df = pd.DataFrame(self.severities_thresholds).T  # (R, K+1)
        adj    = sev_df.loc[anchor_raters, 1:].mean(axis=0)

        self.anchor_thresholds_thresholds[1:] += adj.values
        sev_df.loc[:, 1:] -= adj
        self.anchor_severities_thresholds = {
            rater: sev_df.loc[rater].values for rater in self.raters
        }
        self.anchor_thresholds_thresholds[1:] -= (
            self.anchor_thresholds_thresholds[1:].mean()
        )

    def _calibrate_anchor_matrix(self, anchor_raters):
        '''
        Anchor calibration for matrix parameterisation.
        Subtracts the mean anchor rater severity (per item, per threshold)
        from all raters, and absorbs it into item difficulties and thresholds.
        '''
        self.anchor_diffs_matrix      = self.diffs.copy()
        self.anchor_thresholds_matrix = self.thresholds.copy()

        # Build (R, I, K+1) severity array
        sev_array = np.array([
            [self.severities_matrix[rater][item]
             for item in self.dataframe.columns]
            for rater in self.raters
        ])  # (R, I, K+1)

        # Build (R_anchor, I, K+1) anchor severity array
        anchor_sev_array = np.array([
            [self.severities_matrix[rater][item]
             for item in self.dataframe.columns]
            for rater in anchor_raters
        ])  # (R_anchor, I, K+1)

        # Mean across anchor raters: (I, K+1)
        severity_adjustments = anchor_sev_array.mean(axis=0)

        # Per-item adjustment: mean across thresholds (K slots, skip slot 0)
        diff_adjustments = severity_adjustments[:, 1:].mean(axis=1)  # (I,)

        # Per-threshold adjustment: mean across items
        threshold_adjustments = severity_adjustments[:, 1:].mean(axis=0)  # (K,)

        # Absorb into difficulties and thresholds
        for i, item in enumerate(self.dataframe.columns):
            self.anchor_diffs_matrix[item] += diff_adjustments[i]
        self.anchor_thresholds_matrix[1:] += threshold_adjustments

        # Subtract full adjustment from all raters
        sev_adj = sev_array.copy()
        for r in range(len(self.raters)):
            sev_adj[r, :, :] -= severity_adjustments

        # Re-centre diffs and thresholds (do NOT push back into severities)
        self.anchor_diffs_matrix      -= self.anchor_diffs_matrix.mean()
        self.anchor_thresholds_matrix[1:] -= self.anchor_thresholds_matrix[1:].mean()

        self.anchor_severities_matrix = {
            rater: {
                item: sev_adj[r, j, :]
                for j, item in enumerate(self.dataframe.columns)
            }
            for r, rater in enumerate(self.raters)
        }

        # Marginal severities
        sev_dict = {
            rater: pd.DataFrame(self.anchor_severities_matrix[rater]).iloc[1:]
            for rater in self.raters
        }
        sev_df = pd.concat(sev_dict.values(), keys=sev_dict.keys())

        self.anchor_marginal_severities_items = {
            rater: sev_df.xs(rater).mean(axis=0)
            for rater in self.raters
        }
        self.anchor_marginal_severities_thresholds = {
            rater: pd.concat([pd.Series([0.0]),
                              sev_df.xs(rater).mean(axis=1)])
            for rater in self.raters
        }
        for rater in self.raters:
            adj = self.anchor_marginal_severities_thresholds[rater].iloc[1:].mean()
            self.anchor_marginal_severities_thresholds[rater].iloc[1:] -= adj

    # Backwards-compatible aliases
    def calibrate_global_anchor(self, anchor_raters, **kw):
        self.calibrate_anchor('global', anchor_raters, **kw)
    def calibrate_items_anchor(self, anchor_raters, **kw):
        self.calibrate_anchor('items', anchor_raters, **kw)
    def calibrate_thresholds_anchor(self, anchor_raters, **kw):
        self.calibrate_anchor('thresholds', anchor_raters, **kw)
    def calibrate_matrix_anchor(self, anchor_raters, **kw):
        self.calibrate_anchor('matrix', anchor_raters, **kw)


    # ------------------------------------------------------------------
    # Standard errors (bootstrap)
    # ------------------------------------------------------------------

    def _bootstrap_samples(self, no_of_samples):
        '''Generate bootstrap person samples preserving rater structure.'''
        picks = [
            self.dataframe.index.get_level_values(1)[
                np.random.randint(0, self.no_of_persons, self.no_of_persons)
            ]
            for _ in range(no_of_samples)
        ]
        data_dict = {rater: self.dataframe.xs(rater) for rater in self.raters}
        samples = []
        for pick in picks:
            sample_dict = {
                rater: pd.DataFrame(
                    [data_dict[rater].loc[p] for p in pick]
                ).reset_index(drop=True)
                for rater in self.raters
            }
            samples.append(pd.concat(sample_dict.values(),
                                     keys=sample_dict.keys()))
        return [MFRM(s, self.max_score) for s in samples]

    def _se_from_bootstrap(self, ests_arr, labels, interval):
        '''Compute SE and optional CI from a (B, N) bootstrap array.'''
        se = np.nanstd(ests_arr, axis=0)
        if interval is not None:
            lo = np.percentile(ests_arr, 50 * (1 - interval), axis=0)
            hi = np.percentile(ests_arr, 50 * (1 + interval), axis=0)
        else:
            lo = hi = None
        return se, lo, hi

    def std_errors(self, model='global', anchor_raters=None, interval=None,
                   no_of_samples=100, constant=0.1, method='cos',
                   matrix_power=3, log_lik_tol=0.000001):
        '''
        Bootstrap standard errors for item difficulties, thresholds, and
        rater severities for the specified model.
        '''
        samples = self._bootstrap_samples(no_of_samples)
        for s in samples:
            s.calibrate(model=model, constant=constant, method=method,
                        matrix_power=matrix_power, log_lik_tol=log_lik_tol)
            if anchor_raters is not None:
                s.calibrate_anchor(model, anchor_raters, constant=constant,
                                   method=method, matrix_power=matrix_power,
                                   log_lik_tol=log_lik_tol)

        anc = anchor_raters is not None
        prefix = 'anchor_' if anc else ''

        # Item estimates
        if anc:
            item_ests = np.array([
                getattr(s, f'anchor_diffs_{model}').values for s in samples
            ])
            thresh_ests = np.array([
                getattr(s, f'anchor_thresholds_{model}') for s in samples
            ])
        else:
            item_ests   = np.array([s.diffs.values for s in samples])
            thresh_ests = np.array([s.thresholds for s in samples])

        item_se, item_lo, item_hi = self._se_from_bootstrap(
            item_ests, self.dataframe.columns, interval
        )
        self.item_se = pd.Series(item_se, index=self.dataframe.columns)
        self.item_low  = pd.Series(item_lo, index=self.dataframe.columns) if item_lo is not None else None
        self.item_high = pd.Series(item_hi, index=self.dataframe.columns) if item_hi is not None else None

        thr_se, thr_lo, thr_hi = self._se_from_bootstrap(thresh_ests, None, interval)
        setattr(self, f'{prefix}threshold_se_{model}',   thr_se)
        setattr(self, f'{prefix}threshold_low_{model}',  thr_lo)
        setattr(self, f'{prefix}threshold_high_{model}', thr_hi)

        # Category width SEs
        cat_widths = {
            k + 1: thresh_ests[:, k + 2] - thresh_ests[:, k + 1]
            for k in range(self.max_score - 1)
        }
        setattr(self, f'{prefix}cat_width_se_{model}',
                {k: np.nanstd(v) for k, v in cat_widths.items()})
        if interval is not None:
            setattr(self, f'{prefix}cat_width_low_{model}',
                    {k: np.percentile(v, 50*(1-interval)) for k,v in cat_widths.items()})
            setattr(self, f'{prefix}cat_width_high_{model}',
                    {k: np.percentile(v, 50*(1+interval)) for k,v in cat_widths.items()})

        # Rater SE — structure differs by model
        self._store_rater_se(model, samples, anc, interval, prefix)

    def _store_rater_se(self, model, samples, anchor, interval, prefix):
        '''Store rater SE attributes for the given model.'''
        lo_p = 50 * (1 - interval) if interval is not None else None
        hi_p = 50 * (1 + interval) if interval is not None else None

        if model == 'global':
            sev_attr = f'anchor_severities_global' if anchor else 'severities_global'
            rater_ests = np.array([
                getattr(s, sev_attr).values for s in samples
            ])
            se = pd.Series(np.nanstd(rater_ests, axis=0), index=self.raters)
            setattr(self, f'{prefix}rater_se_{model}', se)
            if interval is not None:
                setattr(self, f'{prefix}rater_low_{model}',
                        pd.Series(np.percentile(rater_ests, lo_p, axis=0), index=self.raters))
                setattr(self, f'{prefix}rater_high_{model}',
                        pd.Series(np.percentile(rater_ests, hi_p, axis=0), index=self.raters))

        elif model == 'items':
            sev_attr = f'anchor_severities_items' if anchor else 'severities_items'
            rater_ests = pd.concat(
                {i: pd.DataFrame.from_dict(getattr(s, sev_attr), orient='index')
                 for i, s in enumerate(samples)},
                keys=range(len(samples))
            ).swaplevel(0, 1)
            se = {rater: rater_ests.xs(rater).std()
                  for rater in self.raters}
            setattr(self, f'{prefix}rater_se_{model}', se)
            if interval is not None:
                setattr(self, f'{prefix}rater_low_{model}',
                        {r: rater_ests.xs(r).quantile(lo_p/100) for r in self.raters})
                setattr(self, f'{prefix}rater_high_{model}',
                        {r: rater_ests.xs(r).quantile(hi_p/100) for r in self.raters})

        elif model == 'thresholds':
            sev_attr = f'anchor_severities_thresholds' if anchor else 'severities_thresholds'
            rater_ests = np.array([
                list(getattr(s, sev_attr).values()) for s in samples
            ])
            se = {rater: rater_ests[:, r, :].std(axis=0)
                  for r, rater in enumerate(self.raters)}
            setattr(self, f'{prefix}rater_se_{model}', se)
            if interval is not None:
                setattr(self, f'{prefix}rater_low_{model}',
                        {rater: np.percentile(rater_ests[:, r, :], lo_p, axis=0)
                         for r, rater in enumerate(self.raters)})
                setattr(self, f'{prefix}rater_high_{model}',
                        {rater: np.percentile(rater_ests[:, r, :], hi_p, axis=0)
                         for r, rater in enumerate(self.raters)})

        elif model == 'matrix':
            sev_attr = f'anchor_severities_matrix' if anchor else 'severities_matrix'
            # Per-(rater, item) SE across bootstrap samples
            by_rater = {
                rater: {
                    i: getattr(s, sev_attr)[rater]
                    for i, s in enumerate(samples)
                }
                for rater in self.raters
            }
            se = {}
            se_marginal_items_all = {}
            se_marginal_thresholds_all = {}
            for rater in self.raters:
                df = pd.DataFrame(by_rater[rater]).T  # (B, I×K+1 dict)
                # se is per item: std of per-item arrays across bootstrap
                item_arrays = {
                    item: np.array([
                        getattr(s, sev_attr)[rater][item]
                        for s in samples
                    ])
                    for item in self.dataframe.columns
                }
                se[rater] = {item: item_arrays[item].std(axis=0)
                             for item in self.dataframe.columns}

                # Marginal SEs computed from bootstrap samples directly
                # Per-item marginal: mean of σ_{r,i,k} across k, std across bootstrap
                se_marginal_items_rater = {
                    item: np.array([
                        getattr(s, sev_attr)[rater][item][1:].mean()
                        for s in samples
                    ]).std()
                    for item in self.dataframe.columns
                }
                # Per-threshold marginal: mean of σ_{r,i,k} across i, std across bootstrap
                thr_marginals = np.array([
                    np.array([
                        getattr(s, sev_attr)[rater][item][1:]
                        for item in self.dataframe.columns
                    ]).mean(axis=0)
                    for s in samples
                ])  # (B, K)
                se_marginal_thresholds_rater = np.concatenate(
                    [[0.0], thr_marginals.std(axis=0)]
                )
                se_marginal_items_all[rater]      = se_marginal_items_rater
                se_marginal_thresholds_all[rater] = se_marginal_thresholds_rater

            setattr(self, f'{prefix}rater_se_{model}', se)
            setattr(self, f'{prefix}rater_se_marginal_items', se_marginal_items_all)
            setattr(self, f'{prefix}rater_se_marginal_thresholds', se_marginal_thresholds_all)

    # Backwards-compatible aliases
    def std_errors_global(self, anchor_raters=None, **kw):
        self.std_errors(model='global', anchor_raters=anchor_raters, **kw)
    def std_errors_items(self, anchor_raters=None, **kw):
        self.std_errors(model='items', anchor_raters=anchor_raters, **kw)
    def std_errors_thresholds(self, anchor_raters=None, **kw):
        self.std_errors(model='thresholds', anchor_raters=anchor_raters, **kw)
    def std_errors_matrix(self, anchor_raters=None, **kw):
        self.std_errors(model='matrix', anchor_raters=anchor_raters, **kw)
    def std_errors_global_anchor(self, anchor_raters, **kw):
        self.std_errors(model='global', anchor_raters=anchor_raters, **kw)

    # ------------------------------------------------------------------
    # Category probability dictionary
    # ------------------------------------------------------------------

    def category_probability_dict(self, model='global', anchor=False,
                                  warm_corr=True, ext_scores=True,
                                  tolerance=0.00001, max_iters=100,
                                  ext_score_adjustment=0.5, method='cos',
                                  constant=0.1, matrix_power=3,
                                  log_lik_tol=0.000001):
        '''Build the (Rater, Person) × Items category probability DataFrames.'''
        difficulties, thresholds, severities = self._get_params(model, anchor)

        if not hasattr(self, f'abils_{model}'):
            self.person_abils(model=model, anchor=anchor, warm_corr=warm_corr,
                              tolerance=tolerance, max_iters=max_iters,
                              ext_score_adjustment=ext_score_adjustment)
        abilities = getattr(self, f'{"anchor_" if anchor else ""}abils_{model}')

        person_filter = self.dataframe.notna().astype(float).replace(0, np.nan)

        if not ext_scores:
            scores     = sum(person_filter.loc[r].sum(axis=1) * self.max_score
                             for r in self.raters)
            total_scores = sum(self.dataframe.loc[r].sum(axis=1)
                               for r in self.raters)
            abilities = abilities[(total_scores > 0) & (total_scores < scores)]
            person_filter = self.dataframe.loc[
                (slice(None), abilities.index), :
            ].notna().astype(float).replace(0, np.nan)

        probs_dict, cats = self._cat_probs_mfrm(
            abilities.values, list(self.items), list(self.raters),
            thresholds, model, severities
        )
        # Convert to per-category (Rater×Person, Items) DataFrames
        cat_prob_dict = {}
        for cat_idx in range(len(cats)):
            frames = {
                rater: pd.DataFrame(
                    probs_dict[rater][cat_idx, :, :],
                    index=abilities.index,
                    columns=self.items
                )
                for rater in self.raters
            }
            df_cat = pd.concat(frames.values(), keys=frames.keys())
            df_cat *= person_filter
            cat_prob_dict[cat_idx] = df_cat

        setattr(self, f'cat_prob_dict_{model}', cat_prob_dict)

    # Backwards-compatible aliases
    def category_probability_dict_global(self, **kw):
        self.category_probability_dict(model='global', **kw)
    def category_probability_dict_items(self, **kw):
        self.category_probability_dict(model='items', **kw)
    def category_probability_dict_thresholds(self, **kw):
        self.category_probability_dict(model='thresholds', **kw)
    def category_probability_dict_matrix(self, **kw):
        self.category_probability_dict(model='matrix', **kw)

    # ------------------------------------------------------------------
    # Ability estimation
    # ------------------------------------------------------------------

    def abil(self, persons, model='global', anchor=False, items=None,
             raters=None, warm_corr=True, tolerance=0.00001,
             max_iters=100, ext_score_adjustment=0.5):
        '''
        Newton-Raphson ML ability estimation with optional Warm correction.

        The key difference between models is how the log-numerator is constructed
        per rater — handled entirely by _cat_probs_mfrm() so the NR loop is
        identical across all four parameterisations.
        '''
        if isinstance(persons, str):
            persons = self.persons if persons == 'all' else [persons]
        if persons is None:
            persons = self.persons
        if isinstance(items, str):
            items = list(self.items) if items == 'all' else [items]
        if items is None:
            items = list(self.items)
        if raters is None:
            raters = list(self.raters)
        elif isinstance(raters, str):
            raters = list(self.raters) if raters == 'all' else [raters]
        if isinstance(raters, pd.core.indexes.base.Index):
            raters = raters.tolist()

        difficulties, thresholds, severities = self._get_params(model, anchor)
        difficulties = difficulties.loc[items]

        person_data   = self.dataframe.loc[pd.IndexSlice[raters, persons], items]
        person_filter = person_data.notna().astype(float).replace(0, np.nan)

        scores = sum(
            person_data.loc[r].sum(axis=1) for r in raters
        ).astype(float)
        ext_scores_vec = sum(
            person_filter.loc[r].sum(axis=1) for r in raters
        ) * self.max_score

        scores[scores == 0]                  = ext_score_adjustment
        scores[scores == ext_scores_vec]    -= ext_score_adjustment

        item_count = sum(person_filter.loc[r].sum(axis=1) for r in raters)
        mean_diffs = (
            sum((person_filter.loc[r] * difficulties.values).sum(axis=1)
                for r in raters) / item_count
        )

        try:
            estimates = pd.Series(
                np.log(scores.values) - np.log((ext_scores_vec - scores).values)
                + mean_diffs.values,
                index=list(persons)
            )

            active = pd.Series(True, index=list(persons))
            iters  = 0

            while active.any() and iters <= max_iters:
                active_idx = estimates.index[active]

                probs_dict, cats = self._cat_probs_mfrm(
                    estimates.loc[active_idx].values,
                    items, raters, thresholds, model, severities
                )

                # Aggregate expected scores and info across raters
                exp_sum  = pd.Series(0.0, index=active_idx)
                info_sum = pd.Series(0.0, index=active_idx)

                for rater in raters:
                    probs = probs_dict[rater]  # (K+1, N_active, I)
                    pf    = person_filter.loc[rater].loc[active_idx].values  # (N, I)

                    exp = (cats[:, None, None] * probs).sum(axis=0) * pf  # (N, I)
                    dev = cats[:, None, None] - exp[None, :, :]
                    inf = (dev ** 2 * probs).sum(axis=0) * pf             # (N, I)

                    exp_sum  += np.nansum(exp, axis=1)
                    info_sum += np.nansum(inf, axis=1)

                changes = ((exp_sum - scores.loc[active_idx]) / info_sum).clip(-1, 1)
                estimates.loc[active_idx] -= changes
                active.loc[active_idx] = abs(changes) > tolerance
                iters += 1

            if iters >= max_iters and active.any():
                n_nc = int(active.sum())
                warnings.warn(
                    f'{n_nc} person(s) did not converge in abil(model={model!r}) '
                    f'and will be set to NaN. Consider increasing max_iters.',
                    UserWarning, stacklevel=2
                )
                estimates[active] = np.nan

            if warm_corr:
                valid = estimates.notna()
                if valid.any():
                    valid_idx = estimates.index[valid]
                    valid_pf  = person_filter.loc[pd.IndexSlice[raters, valid_idx], :]
                    estimates[valid] += self.warm(
                        estimates[valid], items, raters, severities,
                        thresholds, valid_pf, model
                    )

        except Exception as e:
            warnings.warn(f'abil(model={model!r}) failed with exception: {e}. '
                          'Returning NaN for all persons.',
                          UserWarning, stacklevel=2)
            estimates = pd.Series(np.nan, index=list(persons))

        return estimates

    def person_abils(self, model='global', anchor=False, items=None,
                     raters=None, warm_corr=True, tolerance=0.00001,
                     max_iters=100, ext_score_adjustment=0.5):
        '''Estimate abilities for all persons; store as self.abils_{model}.'''
        estimates = self.abil(
            self.persons, model=model, anchor=anchor, items=items,
            raters=raters, warm_corr=warm_corr, tolerance=tolerance,
            max_iters=max_iters, ext_score_adjustment=ext_score_adjustment
        )
        attr = f'{"anchor_" if anchor else ""}abils_{model}'
        setattr(self, attr, estimates)

    # Backwards-compatible aliases
    def abil_global(self, persons, anchor=False, items=None, raters=None, **kw):
        return self.abil(persons, model='global', anchor=anchor, items=items, raters=raters, **kw)
    def abil_items(self, persons, anchor=False, items=None, raters=None, **kw):
        return self.abil(persons, model='items', anchor=anchor, items=items, raters=raters, **kw)
    def abil_thresholds(self, persons, anchor=False, items=None, raters=None, **kw):
        return self.abil(persons, model='thresholds', anchor=anchor, items=items, raters=raters, **kw)
    def abil_matrix(self, persons, anchor=False, items=None, raters=None, **kw):
        return self.abil(persons, model='matrix', anchor=anchor, items=items, raters=raters, **kw)

    def person_abils_global(self, anchor=False, items=None, raters=None, **kw):
        self.person_abils(model='global', anchor=anchor, items=items, raters=raters, **kw)
    def person_abils_items(self, anchor=False, items=None, raters=None, **kw):
        self.person_abils(model='items', anchor=anchor, items=items, raters=raters, **kw)
    def person_abils_thresholds(self, anchor=False, items=None, raters=None, **kw):
        self.person_abils(model='thresholds', anchor=anchor, items=items, raters=raters, **kw)
    def person_abils_matrix(self, anchor=False, items=None, raters=None, **kw):
        self.person_abils(model='matrix', anchor=anchor, items=items, raters=raters, **kw)

    # ------------------------------------------------------------------
    # Warm correction
    # ------------------------------------------------------------------

    def warm(self, abilities, items, raters, severities, thresholds,
             person_filter, model='global'):
        """
        Apply Warm's (1989) weighted maximum likelihood bias correction.

        Computes the MFRM generalisation of the Warm correction, summing over
        all raters and items. The correction is (J1 - J2 + J3) / (2 * I^2)
        where I is total Fisher information and J1, J2, J3 are cubic moment
        terms. Uses the vectorised _cat_probs_mfrm engine.

        Parameters
        ----------
        abilities : pandas.Series
            Current ability estimates, indexed by person.
        items : list
            Item subset to use.
        raters : list
            Rater subset to use.
        severities : Series or dict
            Rater severity parameters (structure depends on model).
        thresholds : array-like
            Rasch-Andrich threshold vector.
        person_filter : pandas.DataFrame
            Binary mask (1.0 = responded, NaN = missing), with (Rater, Person)
            MultiIndex and items as columns.
        model : str, default 'global'
            Rater parameterisation.

        Returns
        -------
        pandas.Series
            Warm bias correction terms indexed by person, to add to ML estimates.
        """
        probs_dict, cats = self._cat_probs_mfrm(
            abilities.values, items, raters, thresholds, model, severities
        )

        part1 = pd.Series(0.0, index=abilities.index)
        part2 = pd.Series(0.0, index=abilities.index)
        part3 = pd.Series(0.0, index=abilities.index)
        info_sum = pd.Series(0.0, index=abilities.index)

        for rater in raters:
            probs = probs_dict[rater]  # (K+1, N, I)
            if isinstance(person_filter.index, pd.MultiIndex):
                pf = person_filter.loc[rater].values
            else:
                pf = person_filter.values

            exp   = (cats[:, None, None] * probs).sum(axis=0) * pf      # (N, I)
            dev   = cats[:, None, None] - exp[None, :, :]
            info  = (dev ** 2 * probs).sum(axis=0) * pf                  # (N, I)
            masked_probs = probs * np.where(np.isnan(pf), 0, pf)[None, :, :]

            part1    += np.nansum((cats[:, None, None] ** 3 * masked_probs).sum(axis=0), axis=1)
            part2    += 3 * np.nansum((info + exp ** 2) * exp, axis=1)
            part3    += 2 * np.nansum(exp ** 3, axis=1)
            info_sum += np.nansum(info, axis=1)

        den = 2 * info_sum ** 2
        warm_corr = (part1 - part2 + part3) / den
        return pd.Series(warm_corr.values, index=abilities.index)

    # Backwards-compatible aliases
    def warm_global(self, abilities, items, raters, severities, pf, **kw):
        return self.warm(abilities, items, raters, severities, self.thresholds, pf, 'global')
    def warm_items(self, abilities, items, raters, severities, pf, **kw):
        return self.warm(abilities, items, raters, severities, self.thresholds, pf, 'items')
    def warm_thresholds(self, abilities, items, raters, severities, pf, **kw):
        thr = kw.get('thresholds', self.thresholds)
        return self.warm(abilities, items, raters, severities, thr, pf, 'thresholds')
    def warm_matrix(self, abilities, items, raters, severities, pf, **kw):
        return self.warm(abilities, items, raters, severities, self.thresholds, pf, 'matrix')

    # ------------------------------------------------------------------
    # CSEM
    # ------------------------------------------------------------------

    def csem(self, model='global', anchor=False, persons=None, abilities=None,
             items=None, raters=None):
        """
        Compute the conditional standard error of measurement.

        Calculates CSEM = 1 / sqrt(I) where I is total Fisher information
        summed across all observed rater-item combinations for each person.
        Uses the vectorised _cat_probs_mfrm engine.

        Parameters
        ----------
        model : str, default 'global'
            Rater parameterisation.
        anchor : bool, default False
            If True, uses anchor-calibrated parameters.
        persons : list or None, default None
            Subset of persons. None uses all persons.
        abilities : pandas.Series or None, default None
            Ability estimates. If None, uses stored abils_{model}.
        items : list or None, default None
            Item subset. None uses all items.
        raters : list or None, default None
            Rater subset. None uses all raters.

        Returns
        -------
        pandas.Series
            CSEM values indexed by person, in logits.
        """
        difficulties, thresholds, severities = self._get_params(model, anchor)

        if abilities is None:
            abilities = self._get_abils(model, anchor)
        if persons is not None:
            abilities = abilities.loc[persons]
        if items is None:
            items = list(self.items)
        if raters is None:
            raters = list(self.raters)

        person_data   = self.dataframe.loc[(raters, abilities.index), items]
        person_filter = person_data.notna().astype(float).replace(0, np.nan)

        probs_dict, cats = self._cat_probs_mfrm(
            abilities.values, items, raters, thresholds, model, severities
        )

        info_sum = pd.Series(0.0, index=abilities.index)
        for rater in raters:
            probs = probs_dict[rater]
            pf    = person_filter.loc[rater].values
            exp   = (cats[:, None, None] * probs).sum(axis=0) * pf
            dev   = cats[:, None, None] - exp[None, :, :]
            info  = (dev ** 2 * probs).sum(axis=0) * pf
            info_sum += np.nansum(info, axis=1)

        return 1.0 / (info_sum ** 0.5)

    # Backwards-compatible aliases
    def csem_global(self, **kw): return self.csem(model='global', **kw)
    def csem_items(self, **kw):  return self.csem(model='items', **kw)
    def csem_thresholds(self, **kw): return self.csem(model='thresholds', **kw)
    def csem_matrix(self, **kw): return self.csem(model='matrix', **kw)

    # ------------------------------------------------------------------
    # Score-to-ability lookup
    # ------------------------------------------------------------------

    def score_abil(self, score, model='global', anchor=False, items=None,
                   raters=None, warm_corr=True, tolerance=0.00001,
                   max_iters=100, ext_score_adjustment=0.5):
        """
        Convert a raw total score to an ability estimate via Newton-Raphson ML.

        Used internally to draw score lines on TCC plots. Sums expected scores
        and information across all specified rater-item combinations using
        scalar exp_score() and variance() methods.

        Parameters
        ----------
        score : int or float
            Raw total score to convert. Extreme scores adjusted by
            ext_score_adjustment.
        model : str, default 'global'
            Rater parameterisation.
        anchor : bool, default False
            If True, uses anchor-calibrated parameters.
        items : list or None, default None
            Item subset. None uses all items.
        raters : list or None, default None
            Rater subset. None uses all raters.
        warm_corr : bool, default True
            If True, applies Warm's (1989) bias correction.
        tolerance : float, default 0.00001
            Newton-Raphson convergence tolerance.
        max_iters : int, default 100
            Maximum Newton-Raphson iterations.
        ext_score_adjustment : float, default 0.5
            Adjustment applied to extreme scores of 0 or maximum.

        Returns
        -------
        float
            Ability estimate in logits.
        """
        difficulties, thresholds, severities = self._get_params(model, anchor)

        if items is None:
            items = list(self.items)
        elif isinstance(items, str):
            items = list(self.items) if items == 'all' else [items]

        if raters is None:
            raters = list(self.raters)
        elif isinstance(raters, str):
            raters = list(self.raters) if raters == 'all' else [raters]

        difficulties = difficulties.loc[items]
        ext_score    = len(items) * len(raters) * self.max_score
        used_score   = float(score)
        if used_score == 0:
            used_score = ext_score_adjustment
        elif used_score == ext_score:
            used_score -= ext_score_adjustment

        estimate = log(used_score) - log(ext_score - used_score) + float(difficulties.mean())
        change, iters = 1.0, 0

        while abs(change) > tolerance and iters <= max_iters:
            result = sum(
                self.exp_score(estimate, item, difficulties, rater,
                               severities, thresholds, model)
                for item in items for rater in raters
            )
            info = sum(
                self.variance(estimate, item, difficulties, rater,
                              severities, thresholds, model)
                for item in items for rater in raters
            )
            change   = max(-1.0, min(1.0, (result - used_score) / info))
            estimate -= change
            iters    += 1

        if warm_corr:
            # Build a minimal single-person MultiIndex person_filter for warm()
            pf_mi = pd.DataFrame(
                1.0,
                index=pd.MultiIndex.from_product(
                    [raters, ['_score_abil_person_']],
                    names=self.dataframe.index.names
                ),
                columns=items
            )
            estimate += float(self.warm(
                pd.Series({'_score_abil_person_': estimate}),
                items, raters, severities, thresholds, pf_mi, model
            ).iloc[0])

        if iters >= max_iters:
            warnings.warn(
                'Maximum iterations reached before convergence in score_abil(). '
                'Returned estimate may be inaccurate.',
                UserWarning, stacklevel=2
            )
        return estimate

    # Backwards-compatible aliases
    def score_abil_global(self, score, anchor=False, items=None, raters=None, **kw):
        return self.score_abil(score, 'global', anchor, items, raters, **kw)
    def score_abil_items(self, score, anchor=False, items=None, raters=None, **kw):
        return self.score_abil(score, 'items', anchor, items, raters, **kw)
    def score_abil_thresholds(self, score, anchor=False, items=None, raters=None, **kw):
        return self.score_abil(score, 'thresholds', anchor, items, raters, **kw)
    def score_abil_matrix(self, score, anchor=False, items=None, raters=None, **kw):
        return self.score_abil(score, 'matrix', anchor, items, raters, **kw)

    def abil_lookup_table(self, model='global', anchor=False, attribute=True,
                          items=None, raters=None, ext_scores=True,
                          warm_corr=True, tolerance=0.00001,
                          max_iters=100, ext_score_adjustment=0.5):
        """
        Build a score-to-ability lookup table for all possible raw scores.

        Estimates the ability corresponding to every possible raw score across
        the specified rater-item combination using Newton-Raphson, and stores
        the result as self.abil_table.

        Parameters
        ----------
        model : str, default 'global'
            Rater parameterisation.
        anchor : bool, default False
            If True, uses anchor-calibrated parameters.
        attribute : bool, default True
            If True, stores result as self.abil_table.
        items : list or None, default None
            Item subset. None uses all items.
        raters : list or None, default None
            Rater subset. None uses all raters.
        ext_scores : bool, default True
            If True, includes extreme scores adjusted by ext_score_adjustment.
        warm_corr : bool, default True
            If True, applies Warm's (1989) bias correction.
        tolerance : float, default 0.00001
            Newton-Raphson convergence tolerance.
        max_iters : int, default 100
            Maximum Newton-Raphson iterations.
        ext_score_adjustment : float, default 0.5
            Adjustment for extreme scores.

        Attributes set (if attribute=True)
        -----------------------------------
        abil_table : pandas.Series
            Ability estimate for each possible raw score, indexed by score.
        """
        if items is None:
            items = list(self.items)
        if raters is None:
            raters = list(self.raters)

        ext_score = len(items) * len(raters) * self.max_score
        if ext_scores:
            scores      = np.arange(ext_score + 1)
            used_scores = scores.astype(float)
            used_scores[0]  += ext_score_adjustment
            used_scores[-1] -= ext_score_adjustment
        else:
            scores      = np.arange(1, ext_score)
            used_scores = scores.astype(float)

        table = pd.Series({
            score: self.score_abil(used_score, model=model, anchor=anchor,
                                   items=items, raters=raters,
                                   warm_corr=warm_corr, tolerance=tolerance,
                                   max_iters=max_iters,
                                   ext_score_adjustment=ext_score_adjustment)
            for score, used_score in zip(scores, used_scores)
        })
        if attribute:
            setattr(self, f'abil_table_{model}', table)
        else:
            return table

    # Backwards-compatible aliases
    def abil_lookup_table_global(self, **kw): self.abil_lookup_table(model='global', **kw)
    def abil_lookup_table_items(self, **kw):  self.abil_lookup_table(model='items', **kw)
    def abil_lookup_table_thresholds(self, **kw): self.abil_lookup_table(model='thresholds', **kw)
    def abil_lookup_table_matrix(self, **kw): self.abil_lookup_table(model='matrix', **kw)


    # ------------------------------------------------------------------
    # Category counts
    # ------------------------------------------------------------------

    def category_counts_item(self, item, rater=None):
        """
        Return response frequency counts for a single item.

        Parameters
        ----------
        item : str
            Item identifier (must be a column in self.dataframe).
        rater : str or None, default None
            If provided, returns counts for that rater only.
            If None, aggregates across all raters.

        Returns
        -------
        pandas.Series
            Count of each response category (0 to max_score), indexed by
            category value. Returns None and prints a message if item or
            rater is invalid.
        """

        if item not in self.dataframe.columns:
            warnings.warn(f'Invalid item name: {item!r}. Returning None.',
                          UserWarning, stacklevel=2)
            return None
        if rater is None:
            return (self.dataframe[item]
                    .value_counts()
                    .reindex(range(self.max_score + 1), fill_value=0)
                    .astype(int))
        if rater not in self.raters:
            warnings.warn(f'Invalid rater name: {rater!r}. Returning None.',
                          UserWarning, stacklevel=2)
            return None
        return (self.dataframe.xs(rater)[item]
                .value_counts()
                .reindex(range(self.max_score + 1), fill_value=0)
                .astype(int))

    def category_counts_df(self):
        """
        Build and store response frequency tables across all items.

        Computes two tables: an overall table aggregated across all raters,
        and a per-rater breakdown. Both include category counts (0 through
        max_score), total valid responses, and missing responses per item.

        Attributes set
        --------------
        category_counts : pandas.DataFrame
            Overall (all-rater) frequency table with items as rows and
            response categories plus Total and Missing as columns.
            A Total row is appended. All values are integers.
        category_counts_raters : pandas.DataFrame
            Per-rater frequency table with a (Rater, Item) MultiIndex.
            Same column structure as category_counts.
        """

        # Overall category counts (across all raters)
        cat_counts = {
            item: {
                score: int(self.category_counts_item(item).get(score, 0))
                for score in range(self.max_score + 1)
            }
            for item in self.items
        }
        df = pd.DataFrame(cat_counts).T.sort_index(axis=1)
        df['Total']   = self.dataframe.count()
        df['Missing'] = self.dataframe.shape[0] - df['Total']
        df.loc['Total'] = df.sum()
        self.category_counts = df.astype(int)

        # Per-rater category counts
        rater_counts = {}
        for rater in self.raters:
            rater_dict = {
                item: {
                    score: int(self.category_counts_item(item, rater).get(score, 0))
                    for score in range(self.max_score + 1)
                }
                for item in self.items
            }
            rdf = pd.DataFrame(rater_dict).T.sort_index(axis=1)
            rdf['Total']   = self.dataframe.xs(rater).count()
            rdf['Missing'] = (len(self.dataframe.xs(rater).index) -
                              rdf['Total'])
            rdf.loc['Total'] = rdf.sum()
            rater_counts[rater] = rdf

        self.category_counts_raters = pd.concat(
            rater_counts.values(), keys=rater_counts.keys()
        ).astype(int)

    # ------------------------------------------------------------------
    # Fit matrices (shared engine)
    # ------------------------------------------------------------------

    def fit_matrices(self, cat_prob_dict):
        '''
        Compute expected scores, info, kurtosis, residuals from cat_prob_dict.
        cat_prob_dict: {cat: (Rater×Person, Items) DataFrame}
        '''
        exp_score_df = sum(cat * df for cat, df in cat_prob_dict.items())
        info_df      = sum(df * (cat - exp_score_df) ** 2
                           for cat, df in cat_prob_dict.items())
        kurtosis_df  = sum(df * (cat - exp_score_df) ** 4
                           for cat, df in cat_prob_dict.items())
        residual_df  = self.dataframe.loc[exp_score_df.index] - exp_score_df
        std_residual_df = residual_df / (info_df ** 0.5)
        return exp_score_df, info_df, kurtosis_df, residual_df, std_residual_df

    def _ensure_fit_matrices(self, model, **kw):
        '''Ensure calibration, abilities, cat_prob_dict and fit matrices exist.'''
        calib_kw = {k: v for k, v in kw.items()
                    if k in ('constant', 'method', 'matrix_power', 'log_lik_tol')}
        abil_kw  = {k: v for k, v in kw.items()
                    if k in ('warm_corr', 'tolerance', 'max_iters', 'ext_score_adjustment')}
        if not hasattr(self, f'severities_{model}'):
            self.calibrate(model=model, **calib_kw)
        if not hasattr(self, f'abils_{model}'):
            self.person_abils(model=model, **abil_kw)
        cpd_attr = f'cat_prob_dict_{model}'
        exp_attr = f'exp_score_df_{model}'
        if not hasattr(self, cpd_attr):
            self.category_probability_dict(model=model, **kw)
        if not hasattr(self, exp_attr):
            cpd = getattr(self, cpd_attr)
            (exp, info, kur, res, std) = self.fit_matrices(cpd)
            setattr(self, f'exp_score_df_{model}',    exp)
            setattr(self, f'info_df_{model}',         info)
            setattr(self, f'kurtosis_df_{model}',     kur)
            setattr(self, f'residual_df_{model}',     res)
            setattr(self, f'std_residual_df_{model}', std)

    def fit_matrices_global(self, **kw):
        self._ensure_fit_matrices('global', **kw)
    def fit_matrices_items(self, **kw):
        self._ensure_fit_matrices('items', **kw)
    def fit_matrices_thresholds(self, **kw):
        self._ensure_fit_matrices('thresholds', **kw)
    def fit_matrices_matrix(self, **kw):
        self._ensure_fit_matrices('matrix', **kw)

    # ------------------------------------------------------------------
    # Item fit statistics
    # ------------------------------------------------------------------

    def item_fit_statistics(self, exp_score_df, info_df, kurtosis_df,
                             residual_df, std_residual_df, abilities):
        '''Shared item fit statistics computation.'''
        scores     = self.dataframe.sum(axis=1)
        max_scores = self.dataframe.count(axis=1) * self.max_score
        item_count = self.dataframe[(scores > 0) & (scores < max_scores)].count(axis=0)
        self.response_counts = self.dataframe.count(axis=0)
        self.item_facilities = self.dataframe.mean(axis=0) / self.max_score

        item_outfit_ms   = (std_residual_df ** 2).mean()
        item_outfit_zstd = (((item_outfit_ms ** (1/3)) - 1
                             + 2 / (9 * item_count))
                            / (2 / (9 * item_count)) ** 0.5)

        item_infit_ms   = (residual_df ** 2).sum() / info_df.sum()
        item_infit_zstd = (((item_infit_ms ** (1/3)) - 1
                             + 2 / (9 * item_count))
                            / (2 / (9 * item_count)) ** 0.5)

        # Expand abilities to (Rater×Person) MultiIndex
        abils_by_rater = pd.concat(
            {rater: abilities for rater in self.raters},
            keys=self.raters
        )
        abils_by_rater.index.names = self.dataframe.index.names
        pm, exp_pm = self.pt_meas(abils_by_rater, exp_score_df, info_df)

        return (item_outfit_ms, item_outfit_zstd, item_infit_ms, item_infit_zstd,
                pm, exp_pm)

    def _run_item_fit(self, model, **kw):
        self._ensure_fit_matrices(model, **kw)
        abilities = getattr(self, f'abils_{model}')
        (outfit_ms, outfit_z, infit_ms, infit_z, pm, exp_pm) = self.item_fit_statistics(
            getattr(self, f'exp_score_df_{model}'),
            getattr(self, f'info_df_{model}'),
            getattr(self, f'kurtosis_df_{model}'),
            getattr(self, f'residual_df_{model}'),
            getattr(self, f'std_residual_df_{model}'),
            abilities
        )
        setattr(self, f'item_outfit_ms_{model}',   outfit_ms)
        setattr(self, f'item_outfit_zstd_{model}', outfit_z)
        setattr(self, f'item_infit_ms_{model}',    infit_ms)
        setattr(self, f'item_infit_zstd_{model}',  infit_z)
        setattr(self, f'point_measure_{model}',     pm)
        setattr(self, f'exp_point_measure_{model}', exp_pm)

    def item_fit_statistics_global(self, **kw):   self._run_item_fit('global', **kw)
    def item_fit_statistics_items(self, **kw):    self._run_item_fit('items', **kw)
    def item_fit_statistics_thresholds(self, **kw): self._run_item_fit('thresholds', **kw)
    def item_fit_statistics_matrix(self, **kw):  self._run_item_fit('matrix', **kw)

    # ------------------------------------------------------------------
    # Threshold fit statistics
    # ------------------------------------------------------------------

    def threshold_fit_statistics(self, abilities, diff_df_dict):
        '''Shared threshold fit statistics (dichotomised ICC approach).
        Mirrors RSM threshold_fit_statistics but with (Rater, Person) MultiIndex
        and nz filter for extreme total scores.
        '''
        # Build (Rater×Person, Items) ability DataFrame
        basic_abils_df = pd.DataFrame(
            [[abilities[person] for _ in self.dataframe.columns]
             for person in self.persons],
            index=self.persons,
            columns=self.dataframe.columns
        )
        abil_df = pd.concat(
            [basic_abils_df] * self.no_of_raters,
            keys=list(self.raters)
        )
        abil_df.index.names = self.dataframe.index.names

        scores     = self.dataframe.sum(axis=1)
        max_scores = self.dataframe.count(axis=1) * self.max_score
        nz         = (scores > 0) & (scores < max_scores)

        dich = {}
        for t in range(self.max_score):
            d = self.dataframe.where(self.dataframe.isin([t, t + 1]), np.nan) - t
            d.index.names = self.dataframe.index.names
            dich[t + 1] = d

        # Count non-missing in raw dich (before nz) — matches RSM
        dich_cnt = {t + 1: dich[t + 1].notna().sum().sum()
                    for t in range(self.max_score)}

        dich_exp = {}
        dich_var = {}
        dich_kur = {}
        dich_res = {}
        dich_std = {}

        for t in range(self.max_score):
            mm = (dich[t + 1] + 1) / (dich[t + 1] + 1)
            mm = mm.loc[nz]
            mm.index.names = self.dataframe.index.names

            p = 1.0 / (1.0 + np.exp(diff_df_dict[t + 1] - abil_df))
            p = p.loc[nz]
            p.index.names = self.dataframe.index.names
            p = p * mm

            v = p * (1 - p) * mm
            k = (((-p) ** 4) * (1 - p) + ((1 - p) ** 4) * p) * mm

            dich_exp[t + 1] = p
            dich_var[t + 1] = v
            dich_kur[t + 1] = k

            d_t = dich[t + 1].loc[nz]
            d_t.index.names = self.dataframe.index.names
            dich_res[t + 1] = d_t - p
            dich_std[t + 1] = dich_res[t + 1] / (v ** 0.5)

        def _series(fn):
            return pd.Series({t + 1: fn(t) for t in range(self.max_score)})

        # Outfit MS: sum(std_res²) / count of valid dich responses (matching RSM)
        outfit_ms = _series(lambda t: (
            (dich_std[t+1] ** 2).sum().sum() / dich[t+1].loc[nz].count().sum()
        ))
        infit_ms  = _series(lambda t: (
            (dich_res[t+1] ** 2).sum().sum() / dich_var[t+1].sum().sum()
            if dich_var[t+1].sum().sum() > 0 else np.nan
        ))

        outfit_q  = (_series(lambda t: (
            (dich_kur[t+1] / dich_var[t+1] ** 2).sum().sum() / dich_cnt[t+1] ** 2
            - 1 / dich_cnt[t+1]
        )) ** 0.5)
        infit_q   = (_series(lambda t: (
            (dich_kur[t+1] - dich_var[t+1] ** 2).sum().sum()
            / dich_var[t+1].sum().sum() ** 2
        )) ** 0.5)

        outfit_z = ((outfit_ms ** (1/3) - 1) * (3 / outfit_q) + outfit_q / 3)
        infit_z  = ((infit_ms  ** (1/3) - 1) * (3 / infit_q)  + infit_q  / 3)

        # Point-measure correlations
        abil_dev = pd.concat(
            [abilities.loc[self.persons] - abilities.loc[self.persons].mean()] * self.no_of_raters,
            keys=list(self.raters)
        ).loc[nz]
        abil_dev.index.names = self.dataframe.index.names

        fac = {t+1: dich[t+1].loc[nz].mean() for t in range(self.max_score)}

        pm_num = _series(lambda t: (
            (dich[t+1].loc[nz] - fac[t+1]).mul(abil_dev.values, axis=0).sum().sum()
        ))
        pm_den = _series(lambda t: (
            ((dich[t+1].loc[nz] - fac[t+1]) ** 2).sum().sum()
            * float((abil_dev ** 2).sum())
        ) ** 0.5)
        thresh_pm = pm_num / pm_den

        exp_pm_c = {t+1: dich_exp[t+1] - dich_exp[t+1].mean()
                    for t in range(self.max_score)}
        exp_pm_num = _series(lambda t: exp_pm_c[t+1].mul(abil_dev.values, axis=0).sum().sum())
        exp_pm_den = _series(lambda t: (
            ((exp_pm_c[t+1] ** 2) + dich_var[t+1]).sum().sum()
            * float((abil_dev ** 2).sum())
        ) ** 0.5)
        thresh_exp_pm = exp_pm_num / exp_pm_den

        # Discrimination
        diff_dev = {}
        for t in range(self.max_score):
            dd = abil_df - diff_df_dict[t+1]
            dd = dd.loc[nz]
            dd.index.names = self.dataframe.index.names
            diff_dev[t+1] = dd

        disc_num = _series(lambda t: (diff_dev[t+1] * dich_res[t+1]).sum().sum())
        disc_den = _series(lambda t: (dich_var[t+1] * diff_dev[t+1] ** 2).sum().sum())
        discrimination = 1 + disc_num / disc_den

        return (outfit_ms, outfit_z, infit_ms, infit_z,
                thresh_pm, thresh_exp_pm, discrimination)


    def _diff_df_dict(self, model, difficulties, thresholds, severities):
        '''Build the threshold location DataFrame dict for threshold fit stats.'''
        diff_df_dict = {}
        for t in range(self.max_score):
            thr_loc = thresholds[t + 1]
            rows = {}
            for rater in self.raters:
                if model == 'global':
                    row = difficulties + thr_loc + float(severities.loc[rater])
                elif model == 'items':
                    sev_series = pd.Series(severities[rater]).reindex(self.dataframe.columns)
                    row = difficulties + thr_loc + sev_series
                elif model == 'thresholds':
                    row = difficulties + thr_loc + severities[rater][t + 1]
                elif model == 'matrix':
                    row = difficulties + thr_loc + pd.Series({
                        item: severities[rater][item][t + 1]
                        for item in self.dataframe.columns
                    })
                rows[rater] = pd.DataFrame(
                    np.tile(row.values[None, :], (self.no_of_persons, 1)),
                    index=self.persons, columns=self.dataframe.columns
                )
            df_t = pd.concat(list(rows.values()), keys=list(rows.keys()))
            df_t.index.names = self.dataframe.index.names
            diff_df_dict[t + 1] = df_t
        return diff_df_dict

    def _run_threshold_fit(self, model, anchor_raters=None, **kw):
        if not hasattr(self, f'abils_{model}'):
            self.person_abils(model=model)
        # Always use unanchored params for fit statistics — anchor is origin shift only
        difficulties, thresholds, severities = self._get_params(model, anchor=False)
        abilities  = getattr(self, f'abils_{model}')
        ddd        = self._diff_df_dict(model, difficulties, thresholds, severities)
        results    = self.threshold_fit_statistics(abilities, ddd)
        names      = ['threshold_outfit_ms', 'threshold_outfit_zstd',
                      'threshold_infit_ms',  'threshold_infit_zstd',
                      'threshold_point_measure', 'threshold_exp_point_measure',
                      'threshold_discrimination']
        for name, val in zip(names, results):
            setattr(self, f'{name}_{model}', val)

    def threshold_fit_statistics_global(self, **kw):    self._run_threshold_fit('global', **kw)
    def threshold_fit_statistics_items(self, **kw):     self._run_threshold_fit('items', **kw)
    def threshold_fit_statistics_thresholds(self, **kw): self._run_threshold_fit('thresholds', **kw)
    def threshold_fit_statistics_matrix(self, **kw):    self._run_threshold_fit('matrix', **kw)

    # ------------------------------------------------------------------
    # Rater fit statistics
    # ------------------------------------------------------------------

    def rater_pivot(self, df):
        '''Pivot (Rater×Person, Items) DataFrame to (Person×Items, Raters).'''
        return pd.DataFrame({
            rater: df.xs(rater).T.stack()
            for rater in self.raters
        })

    def rater_fit_statistics(self, info_df, kurtosis_df, residual_df,
                              std_residual_df):
        '''Shared rater fit statistics.'''
        scores     = self.dataframe.sum(axis=1)
        max_scores = self.dataframe.count(axis=1) * self.max_score
        rater_count = pd.Series({
            rater: self.dataframe[(scores > 0) & (scores < max_scores)
                   ].xs(rater).count().sum()
            for rater in self.raters
        })

        rater_outfit_ms = pd.Series({
            rater: ((std_residual_df ** 2).xs(rater).sum().sum() /
                    (std_residual_df ** 2).xs(rater).count().sum())
            for rater in self.raters
        })
        rater_infit_ms = pd.Series({
            rater: ((residual_df ** 2).xs(rater).sum().sum() /
                    info_df.xs(rater).sum().sum())
            for rater in self.raters
        })

        rater_outfit_q = (
            (self.rater_pivot(kurtosis_df) / (self.rater_pivot(info_df) ** 2))
            / (rater_count ** 2)
        ).sum() - 1 / rater_count
        rater_outfit_q = rater_outfit_q ** 0.5

        rater_outfit_zstd = (((rater_outfit_ms ** (1/3)) - 1) *
                              (3 / rater_outfit_q) + rater_outfit_q / 3)

        rater_infit_q = (
            (self.rater_pivot(kurtosis_df) - self.rater_pivot(info_df) ** 2).sum() /
            (self.rater_pivot(info_df).sum() ** 2)
        ) ** 0.5
        rater_infit_zstd = (((rater_infit_ms ** (1/3)) - 1) *
                             (3 / rater_infit_q) + rater_infit_q / 3)

        return rater_outfit_ms, rater_outfit_zstd, rater_infit_ms, rater_infit_zstd

    def _run_rater_fit(self, model, **kw):
        self._ensure_fit_matrices(model, **kw)
        results = self.rater_fit_statistics(
            getattr(self, f'info_df_{model}'),
            getattr(self, f'kurtosis_df_{model}'),
            getattr(self, f'residual_df_{model}'),
            getattr(self, f'std_residual_df_{model}')
        )
        for name, val in zip(
            ['rater_outfit_ms', 'rater_outfit_zstd',
             'rater_infit_ms',  'rater_infit_zstd'],
            results
        ):
            setattr(self, f'{name}_{model}', val)

    def rater_fit_statistics_global(self, **kw):     self._run_rater_fit('global', **kw)
    def rater_fit_statistics_items(self, **kw):      self._run_rater_fit('items', **kw)
    def rater_fit_statistics_thresholds(self, **kw): self._run_rater_fit('thresholds', **kw)
    def rater_fit_statistics_matrix(self, **kw):     self._run_rater_fit('matrix', **kw)

    # ------------------------------------------------------------------
    # Person fit statistics
    # ------------------------------------------------------------------

    def person_fit_statistics(self, info_df, kurtosis_df, residual_df,
                               std_residual_df, abilities, **kw):
        '''Shared person fit statistics.'''
        csems = 1.0 / (info_df.unstack(level=0).sum(axis=1) ** 0.5)
        rsems = (((residual_df.unstack(level=0) ** 2).sum(axis=1)) ** 0.5
                 / info_df.unstack(level=0).sum(axis=1))

        person_outfit_ms = (std_residual_df.unstack(level=0) ** 2).mean(axis=1)
        person_infit_ms  = ((residual_df.unstack(level=0) ** 2).sum(axis=1) /
                             info_df.unstack(level=0).sum(axis=1))

        scores     = self.dataframe.sum(axis=1)
        max_scores = self.dataframe.count(axis=1) * self.max_score
        person_count = (
            self.dataframe[(scores > 0) & (scores < max_scores)]
            .unstack(level=0).notna().sum(axis=1)
        )

        base_df = (kurtosis_df.unstack(level=0) /
                   (info_df.unstack(level=0) ** 2))
        # Sum kurtosis/info² per person, divide by person_count²
        # Avoid the fragile transpose trick — align directly on person index
        base_df = base_df.loc[person_count.index]
        outfit_q_sq = (base_df.sum(axis=1) / (person_count ** 2)) - (1 / person_count)
        person_outfit_q = np.where(outfit_q_sq >= 0, outfit_q_sq ** 0.5, np.nan)
        person_outfit_q = pd.Series(person_outfit_q, index=person_count.index)
        person_outfit_zstd = (((person_outfit_ms ** (1/3)) - 1) *
                               (3 / person_outfit_q) + person_outfit_q / 3)
        person_outfit_zstd = person_outfit_zstd[:self.no_of_persons].astype(float)

        infit_q_sq = ((kurtosis_df.unstack(level=0) -
                       info_df.unstack(level=0) ** 2).sum(axis=1) /
                      (info_df.unstack(level=0).sum(axis=1) ** 2))
        person_infit_q = np.where(infit_q_sq >= 0, infit_q_sq ** 0.5, np.nan)
        person_infit_q = pd.Series(person_infit_q, index=infit_q_sq.index)
        person_infit_zstd = (((person_infit_ms ** (1/3)) - 1) *
                              (3 / person_infit_q) + person_infit_q / 3).astype(float)

        return (csems, rsems, person_outfit_ms, person_outfit_zstd,
                person_infit_ms, person_infit_zstd)

    def _run_person_fit(self, model, **kw):
        self._ensure_fit_matrices(model, **kw)
        abilities = getattr(self, f'abils_{model}')
        results   = self.person_fit_statistics(
            getattr(self, f'info_df_{model}'),
            getattr(self, f'kurtosis_df_{model}'),
            getattr(self, f'residual_df_{model}'),
            getattr(self, f'std_residual_df_{model}'),
            abilities
        )
        names = ['csem_vector', 'rsem_vector', 'person_outfit_ms',
                 'person_outfit_zstd', 'person_infit_ms', 'person_infit_zstd']
        for name, val in zip(names, results):
            if isinstance(val, pd.Series):
                val = pd.to_numeric(val, errors='coerce')
            setattr(self, f'{name}_{model}', val)

    def person_fit_statistics_global(self, **kw):     self._run_person_fit('global', **kw)
    def person_fit_statistics_items(self, **kw):      self._run_person_fit('items', **kw)
    def person_fit_statistics_thresholds(self, **kw): self._run_person_fit('thresholds', **kw)
    def person_fit_statistics_matrix(self, **kw):     self._run_person_fit('matrix', **kw)

    # ------------------------------------------------------------------
    # Test-level fit statistics
    # ------------------------------------------------------------------

    def test_fit_statistics(self, abilities, rsems):
        '''Shared test-level separation and reliability statistics.'''
        scores     = self.dataframe.unstack(level=0).sum(axis=1)
        max_scores = self.dataframe.unstack(level=0).count(axis=1) * self.max_score
        abilities  = abilities[(scores > 0) & (scores < max_scores)]

        isi              = (self.diffs.var() / (self.item_se ** 2).mean() - 1) ** 0.5
        item_strata      = (4 * isi + 1) / 3
        item_reliability = isi ** 2 / (1 + isi ** 2)

        mean_rsem2 = (rsems ** 2).mean()
        psi              = ((np.var(abilities) - mean_rsem2) / mean_rsem2) ** 0.5
        person_strata    = (4 * psi + 1) / 3
        person_reliability = psi ** 2 / (1 + psi ** 2)

        return (isi, item_strata, item_reliability,
                psi, person_strata, person_reliability)

    def _run_test_fit(self, model, **kw):
        if not hasattr(self, f'csem_vector_{model}'):
            self._run_person_fit(model, **kw)
        if not hasattr(self, 'item_se'):
            self.std_errors(model=model, **kw)
        abilities = getattr(self, f'abils_{model}')
        rsems     = getattr(self, f'rsem_vector_{model}')
        results   = self.test_fit_statistics(abilities, rsems)
        for name, val in zip(
            ['isi', 'item_strata', 'item_reliability',
             'psi', 'person_strata', 'person_reliability'],
            results
        ):
            setattr(self, f'{name}_{model}', val)

    def test_fit_statistics_global(self, **kw):     self._run_test_fit('global', **kw)
    def test_fit_statistics_items(self, **kw):      self._run_test_fit('items', **kw)
    def test_fit_statistics_thresholds(self, **kw): self._run_test_fit('thresholds', **kw)
    def test_fit_statistics_matrix(self, **kw):     self._run_test_fit('matrix', **kw)

    # ------------------------------------------------------------------
    # Top-level fit_statistics
    # ------------------------------------------------------------------

    def fit_statistics(self, model='global', anchor_raters=None,
                       warm_corr=True, se=True, test_stats=True,
                       ext_scores=True, tolerance=0.00001, max_iters=100,
                       ext_score_adjustment=0.5, method='cos',
                       constant=0.1, matrix_power=3, log_lik_tol=0.000001,
                       no_of_samples=100, interval=None):
        """
        Compute all item, threshold, rater, person, and test-level fit statistics.

        Top-level orchestrator that auto-triggers calibrate(), std_errors(),
        person_abils(), and category_probability_dict() as needed, then runs
        all fit statistic sub-routines for the specified model. Stores all
        results as model-suffixed attributes.

        Parameters
        ----------
        model : str, default 'global'
            Rater parameterisation: 'global', 'items', 'thresholds', or 'matrix'.
        anchor_raters : list or None, default None
            Rater identifiers to treat as anchors for SE computation.
        warm_corr : bool, default True
            Warm bias correction for ability estimates.
        se : bool, default True
            If True, computes bootstrap SEs. Required for test-level stats.
        test_stats : bool, default True
            If True, computes ISI, PSI, strata, and reliability.
        ext_scores : bool, default True
            If True, includes extreme scorers in category probability dict.
        tolerance : float, default 0.00001
            Newton-Raphson convergence tolerance.
        max_iters : int, default 100
            Maximum Newton-Raphson iterations.
        ext_score_adjustment : float, default 0.5
            Extreme score adjustment.
        method : str, default 'cos'
            Priority vector extraction method for calibration.
        constant : float, default 0.1
            Additive smoothing constant for calibration.
        matrix_power : int, default 3
            Matrix power for calibration.
        log_lik_tol : float, default 0.000001
            Convergence tolerance for calibration.
        no_of_samples : int, default 100
            Bootstrap samples for SE estimation.
        interval : float or None, default None
            CI width for bootstrap estimates.

        Attributes set (model-suffixed)
        --------------------------------
        exp_score_df_{model}, info_df_{model}, kurtosis_df_{model} : DataFrame
            Expected scores, Fisher information, fourth moments.
        residual_df_{model}, std_residual_df_{model} : DataFrame
            Raw and standardised residuals.
        item_infit_ms_{model}, item_outfit_ms_{model} : Series
            Item infit and outfit mean-square.
        item_infit_zstd_{model}, item_outfit_zstd_{model} : Series
            Item Z statistics.
        threshold_infit_ms_{model}, threshold_outfit_ms_{model} : Series
            Threshold infit and outfit mean-square.
        rater_infit_ms_{model}, rater_outfit_ms_{model} : Series
            Rater infit and outfit mean-square.
        person_infit_ms_{model}, person_outfit_ms_{model} : Series
            Person infit and outfit mean-square.
        csem_vector_{model}, rsem_vector_{model} : Series
            Conditional and residual SEM per person.
        isi_{model}, item_strata_{model}, item_reliability_{model} : float
            Item separation index, strata, and reliability (if test_stats).
        psi_{model}, person_strata_{model}, person_reliability_{model} : float
            Person separation index, strata, and reliability (if test_stats).
        """
        if not hasattr(self, f'severities_{model}'):
            self.calibrate(model=model, constant=constant, method=method,
                           matrix_power=matrix_power, log_lik_tol=log_lik_tol)
        if se and not hasattr(self, f'threshold_se_{model}'):
            self.std_errors(model=model, anchor_raters=anchor_raters,
                            interval=interval, no_of_samples=no_of_samples,
                            constant=constant, method=method,
                            matrix_power=matrix_power, log_lik_tol=log_lik_tol)
        if not hasattr(self, f'abils_{model}'):
            self.person_abils(model=model, warm_corr=warm_corr,
                              tolerance=tolerance, max_iters=max_iters,
                              ext_score_adjustment=ext_score_adjustment)
        if not se:
            test_stats = False

        self.category_probability_dict(model=model, warm_corr=warm_corr,
                                       ext_scores=ext_scores,
                                       tolerance=tolerance,
                                       max_iters=max_iters,
                                       ext_score_adjustment=ext_score_adjustment,
                                       method=method, constant=constant,
                                       matrix_power=matrix_power,
                                       log_lik_tol=log_lik_tol)
        self._ensure_fit_matrices(model)
        self._run_item_fit(model)
        self._run_threshold_fit(model, anchor_raters=anchor_raters)
        self._run_rater_fit(model)
        self._run_person_fit(model)
        if test_stats:
            self._run_test_fit(model)

    # Backwards-compatible aliases
    def fit_statistics_global(self, **kw):     self.fit_statistics(model='global', **kw)
    def fit_statistics_items(self, **kw):      self.fit_statistics(model='items', **kw)
    def fit_statistics_thresholds(self, **kw): self.fit_statistics(model='thresholds', **kw)
    def fit_statistics_matrix(self, **kw):     self.fit_statistics(model='matrix', **kw)


    # ------------------------------------------------------------------
    # Residual correlation analysis
    # ------------------------------------------------------------------

    def item_res_corr_analysis(self, std_residual_df):
        """
        Analyse item standardised residual correlations.

        Computes the inter-item correlation matrix of standardised residuals
        and performs PCA to detect violations of local item independence.

        Parameters
        ----------
        std_residual_df : pandas.DataFrame
            Standardised residuals with (Rater, Person) MultiIndex and
            items as columns.

        Returns
        -------
        tuple of (correlations, eigenvectors, eigenvalues, variance_explained, loadings)
            All are DataFrames (or None if PCA fails).
        """
        item_residual_correlations = std_residual_df.corr(numeric_only=False)
        pca = PCA()
        try:
            pca.fit(item_residual_correlations)
            n = self.no_of_items
            pc_labels  = [f'PC {pc + 1}' for pc in range(n)]
            eigvec_labels = [f'Eigenvector {pc + 1}' for pc in range(n)]
            eigenvectors = pd.DataFrame(pca.components_, columns=eigvec_labels)
            eigenvalues  = pd.DataFrame(
                pca.explained_variance_, index=pc_labels, columns=['Eigenvalue']
            )
            variance_explained = pd.DataFrame(
                pca.explained_variance_ratio_,
                index=pc_labels, columns=['Variance explained']
            )
            loadings = pd.DataFrame(
                eigenvectors.values.T * (pca.explained_variance_ ** 0.5),
                index=self.dataframe.columns, columns=pc_labels
            )
        except Exception:
            warnings.warn('PCA of item standardised residuals failed. '
                          'Eigenvectors and loadings set to None.',
                          UserWarning, stacklevel=2)
            eigenvectors = eigenvalues = variance_explained = loadings = None
        return (item_residual_correlations, eigenvectors, eigenvalues,
                variance_explained, loadings)

    def rater_res_corr_analysis(self, residual_df, std_residual_df):
        """
        Analyse rater residual correlations.

        Pivots the residual DataFrame to (Person×Items, Raters) shape,
        computes the inter-rater correlation matrix, and performs PCA.
        A large first eigenvalue suggests systematic rater bias.

        Parameters
        ----------
        residual_df : pandas.DataFrame
            Raw residuals with (Rater, Person) MultiIndex.
        std_residual_df : pandas.DataFrame
            Standardised residuals with (Rater, Person) MultiIndex.

        Returns
        -------
        tuple of (correlations, eigenvectors, eigenvalues, variance_explained, loadings)
            All are DataFrames (or None if PCA fails).
        """
        rater_res     = self.rater_pivot(residual_df)
        rater_std_res = self.rater_pivot(std_residual_df)
        correlations  = rater_res.corr(numeric_only=False)
        pca = PCA()
        try:
            pca.fit(rater_std_res.corr(numeric_only=False))
            n = self.no_of_raters
            pc_labels     = [f'PC {pc + 1}' for pc in range(n)]
            eigvec_labels = [f'Eigenvector {pc + 1}' for pc in range(n)]
            eigenvectors = pd.DataFrame(pca.components_, columns=eigvec_labels)
            eigenvalues  = pd.DataFrame(
                pca.explained_variance_, index=pc_labels, columns=['Eigenvalue']
            )
            variance_explained = pd.DataFrame(
                pca.explained_variance_ratio_,
                index=pc_labels, columns=['Variance explained']
            )
            loadings = pd.DataFrame(
                eigenvectors.values.T * (pca.explained_variance_ ** 0.5),
                index=self.raters, columns=pc_labels
            )
        except Exception:
            warnings.warn('PCA of rater standardised residuals failed. '
                          'Eigenvectors and loadings set to None.',
                          UserWarning, stacklevel=2)
            eigenvectors = eigenvalues = variance_explained = loadings = None
        return (correlations, eigenvectors, eigenvalues,
                variance_explained, loadings)

    def _run_item_res_corr(self, model, **kw):
        if not hasattr(self, f'std_residual_df_{model}'):
            self.fit_statistics(model=model, **kw)
        results = self.item_res_corr_analysis(
            getattr(self, f'std_residual_df_{model}')
        )
        for name, val in zip(
            ['item_residual_correlations', 'item_eigenvectors',
             'item_eigenvalues', 'item_variance_explained', 'item_loadings'],
            results
        ):
            setattr(self, f'{name}_{model}', val)

    def _run_rater_res_corr(self, model, **kw):
        if not hasattr(self, f'std_residual_df_{model}'):
            self.fit_statistics(model=model, **kw)
        results = self.rater_res_corr_analysis(
            getattr(self, f'residual_df_{model}'),
            getattr(self, f'std_residual_df_{model}')
        )
        for name, val in zip(
            ['rater_residual_correlations', 'rater_eigenvectors',
             'rater_eigenvalues', 'rater_variance_explained', 'rater_loadings'],
            results
        ):
            setattr(self, f'{name}_{model}', val)

    def item_res_corr_analysis_global(self, **kw):     self._run_item_res_corr('global', **kw)
    def item_res_corr_analysis_items(self, **kw):      self._run_item_res_corr('items', **kw)
    def item_res_corr_analysis_thresholds(self, **kw): self._run_item_res_corr('thresholds', **kw)
    def item_res_corr_analysis_matrix(self, **kw):     self._run_item_res_corr('matrix', **kw)

    def rater_res_corr_analysis_global(self, **kw):     self._run_rater_res_corr('global', **kw)
    def rater_res_corr_analysis_items(self, **kw):      self._run_rater_res_corr('items', **kw)
    def rater_res_corr_analysis_thresholds(self, **kw): self._run_rater_res_corr('thresholds', **kw)
    def rater_res_corr_analysis_matrix(self, **kw):     self._run_rater_res_corr('matrix', **kw)

    # ------------------------------------------------------------------
    # Output tables
    # ------------------------------------------------------------------

    def _ensure_calibrated(self, model, **kw):
        '''Lazy-load the full chain: calibrate → abils → SE → fit matrices.'''
        calib_kw = {k: v for k, v in kw.items()
                    if k in ('constant', 'method', 'matrix_power', 'log_lik_tol')}
        abil_kw  = {k: v for k, v in kw.items()
                    if k in ('warm_corr', 'tolerance', 'max_iters', 'ext_score_adjustment')}
        se_kw    = {k: v for k, v in kw.items()
                    if k in ('constant', 'method', 'matrix_power', 'log_lik_tol',
                             'no_of_samples', 'interval')}
        anchor_raters = kw.get('anchor_raters', None)

        if not hasattr(self, f'severities_{model}'):
            self.calibrate(model=model, **calib_kw)
        if anchor_raters is not None:
            if not hasattr(self, f'anchor_severities_{model}'):
                self.calibrate_anchor(model, anchor_raters, **calib_kw)
        if not hasattr(self, f'abils_{model}'):
            self.person_abils(model=model, **abil_kw)
        if not hasattr(self, f'threshold_se_{model}'):
            self.std_errors(model=model, anchor_raters=anchor_raters, **se_kw)

    def _ensure_se(self, model, anchor_raters, interval, no_of_samples,
                   constant, method, matrix_power, log_lik_tol):
        prefix = 'anchor_' if anchor_raters is not None else ''
        trigger = f'{prefix}threshold_se_{model}'
        if not hasattr(self, trigger):
            self.std_errors(model=model, anchor_raters=anchor_raters,
                            interval=interval, no_of_samples=no_of_samples,
                            constant=constant, method=method,
                            matrix_power=matrix_power, log_lik_tol=log_lik_tol)

    def item_stats_df(self, model='global', anchor_raters=None, full=False,
                      ext_scores=True, zstd=False, point_measure_corr=False,
                      dp=3, warm_corr=True, tolerance=0.00001, max_iters=100,
                      ext_score_adjustment=0.5, method='cos', constant=0.1,
                      matrix_power=3, log_lik_tol=0.000001, no_of_samples=100,
                      interval=None):
        """
        Build and store the item statistics summary table.

        Auto-triggers the full calibration/SE/fit chain if not yet run.
        Stores result as self.item_stats_{model}.

        Parameters
        ----------
        model : str, default 'global'
            Rater parameterisation.
        anchor_raters : list or None, default None
            If provided, uses anchor-calibrated item difficulties.
        full : bool, default False
            If True, sets zstd=True, point_measure_corr=True, interval=0.95.
        ext_scores : bool, default True
            Include extreme scorers in fit computation.
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
            Additive smoothing constant.
        matrix_power : int, default 3
            Matrix power for calibration.
        log_lik_tol : float, default 0.000001
            Calibration convergence tolerance.
        no_of_samples : int, default 100
            Bootstrap samples for SE estimation.
        interval : float or None, default None
            CI width; if provided, percentile bound columns included.

        Attributes set
        --------------
        item_stats_{model} : pandas.DataFrame
            Item statistics with items as rows. Always contains Estimate,
            SE, Count, Facility, Infit MS, Outfit MS.
        """

        if full:
            zstd = point_measure_corr = True
            interval = interval or 0.95

        self._ensure_calibrated(model, anchor_raters=anchor_raters, interval=interval,
                                no_of_samples=no_of_samples, constant=constant,
                                method=method, matrix_power=matrix_power,
                                log_lik_tol=log_lik_tol, warm_corr=warm_corr,
                                tolerance=tolerance, max_iters=max_iters,
                                ext_score_adjustment=ext_score_adjustment)
        self._ensure_se(model, anchor_raters, interval, no_of_samples,
                        constant, method, matrix_power, log_lik_tol)
        if not hasattr(self, f'item_outfit_ms_{model}'):
            self._run_item_fit(model)

        anc = anchor_raters is not None
        difficulties = (getattr(self, f'anchor_diffs_{model}') if anc
                        else self.diffs)
        se   = self.item_se
        low  = self.item_low
        high = self.item_high

        stats = pd.DataFrame(index=self.dataframe.columns)
        stats['Estimate'] = difficulties.round(dp)
        stats['SE']       = se.round(dp)
        if interval is not None and low is not None:
            lo_lbl = f'{round((1 - interval) * 50, 1)}%'
            hi_lbl = f'{round((1 + interval) * 50, 1)}%'
            stats[lo_lbl] = low.round(dp)
            stats[hi_lbl] = high.round(dp)
        stats['Count']    = self.response_counts.astype(int)
        stats['Facility'] = self.item_facilities.round(dp)
        stats['Infit MS'] = getattr(self, f'item_infit_ms_{model}').round(dp)
        if zstd:
            stats['Infit Z'] = getattr(self, f'item_infit_zstd_{model}').round(dp)
        stats['Outfit MS'] = getattr(self, f'item_outfit_ms_{model}').round(dp)
        if zstd:
            stats['Outfit Z'] = getattr(self, f'item_outfit_zstd_{model}').round(dp)
        if point_measure_corr:
            stats['PM corr']     = getattr(self, f'point_measure_{model}').round(dp)
            stats['Exp PM corr'] = getattr(self, f'exp_point_measure_{model}').round(dp)

        setattr(self, f'item_stats_{model}', stats)

    # Backwards-compatible aliases
    def item_stats_df_global(self, **kw):     self.item_stats_df(model='global', **kw)
    def item_stats_df_items(self, **kw):      self.item_stats_df(model='items', **kw)
    def item_stats_df_thresholds(self, **kw): self.item_stats_df(model='thresholds', **kw)
    def item_stats_df_matrix(self, **kw):     self.item_stats_df(model='matrix', **kw)

    def threshold_stats_df(self, model='global', anchor_raters=None,
                           full=False, zstd=False, disc=False,
                           point_measure_corr=False, dp=3, warm_corr=True,
                           tolerance=0.00001, max_iters=100,
                           ext_score_adjustment=0.5, method='cos',
                           constant=0.1, matrix_power=3, log_lik_tol=0.000001,
                           no_of_samples=100, interval=None):
        """
        Build and store the threshold statistics summary table.

        Auto-triggers the full calibration/SE/fit chain if not yet run.
        Stores result as self.threshold_stats_{model}.

        Parameters
        ----------
        model : str, default 'global'
            Rater parameterisation.
        anchor_raters : list or None, default None
            Anchor raters for SE computation.
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
        matrix_power : int, default 3
            Matrix power.
        log_lik_tol : float, default 0.000001
            Calibration convergence tolerance.
        no_of_samples : int, default 100
            Bootstrap samples.
        interval : float or None, default None
            CI width.

        Attributes set
        --------------
        threshold_stats_{model} : pandas.DataFrame
            Threshold statistics, rows Threshold 1..Threshold max_score.
        """

        if full:
            zstd = disc = point_measure_corr = True
            interval = interval or 0.95

        self._ensure_calibrated(model, anchor_raters=anchor_raters, interval=interval,
                                no_of_samples=no_of_samples, constant=constant,
                                method=method, matrix_power=matrix_power,
                                log_lik_tol=log_lik_tol, warm_corr=warm_corr,
                                tolerance=tolerance, max_iters=max_iters,
                                ext_score_adjustment=ext_score_adjustment)
        self._ensure_se(model, anchor_raters, interval, no_of_samples,
                        constant, method, matrix_power, log_lik_tol)
        if not hasattr(self, f'threshold_outfit_ms_{model}'):
            self._run_threshold_fit(model, anchor_raters=anchor_raters)

        anc        = anchor_raters is not None
        thresholds = (getattr(self, f'anchor_thresholds_{model}') if anc
                      else self.thresholds)
        thr_se_attr = f'anchor_threshold_se_{model}' if anc else f'threshold_se_{model}'
        thr_se  = getattr(self, thr_se_attr, None)
        thr_lo  = getattr(self, f'anchor_threshold_low_{model}'  if anc else f'threshold_low_{model}',  None)
        thr_hi  = getattr(self, f'anchor_threshold_high_{model}' if anc else f'threshold_high_{model}', None)

        idx    = [f'Threshold {t + 1}' for t in range(self.max_score)]
        stats  = pd.DataFrame(index=idx)
        stats['Estimate'] = thresholds[1:].round(dp)
        if thr_se is not None:
            stats['SE'] = thr_se[1:].round(dp)
        if interval is not None and thr_lo is not None:
            lo_lbl = f'{round((1 - interval) * 50, 1)}%'
            hi_lbl = f'{round((1 + interval) * 50, 1)}%'
            stats[lo_lbl] = thr_lo[1:].round(dp)
            stats[hi_lbl] = thr_hi[1:].round(dp)
        stats['Infit MS']  = getattr(self, f'threshold_infit_ms_{model}').values.round(dp)
        if zstd:
            stats['Infit Z']  = getattr(self, f'threshold_infit_zstd_{model}').values.round(dp)
        stats['Outfit MS'] = getattr(self, f'threshold_outfit_ms_{model}').values.round(dp)
        if zstd:
            stats['Outfit Z'] = getattr(self, f'threshold_outfit_zstd_{model}').values.round(dp)
        if disc:
            stats['Discrim'] = getattr(self, f'threshold_discrimination_{model}').values.round(dp)
        if point_measure_corr:
            stats['PM corr']     = getattr(self, f'threshold_point_measure_{model}').values.round(dp)
            stats['Exp PM corr'] = getattr(self, f'threshold_exp_point_measure_{model}').values.round(dp)

        setattr(self, f'threshold_stats_{model}', stats)

    def threshold_stats_df_global(self, **kw):     self.threshold_stats_df(model='global', **kw)
    def threshold_stats_df_items(self, **kw):      self.threshold_stats_df(model='items', **kw)
    def threshold_stats_df_thresholds(self, **kw): self.threshold_stats_df(model='thresholds', **kw)
    def threshold_stats_df_matrix(self, **kw):     self.threshold_stats_df(model='matrix', **kw)

    def person_stats_df(self, model='global', anchor_raters=None,
                        full=False, rsem=False, zstd=False, dp=3,
                        warm_corr=True, tolerance=0.00001, max_iters=100,
                        ext_score_adjustment=0.5, method='cos',
                        constant=0.1, matrix_power=3, log_lik_tol=0.000001,
                        interval=None, no_of_samples=100):
        """
        Build and store the person statistics summary table.

        Auto-triggers calibration and person ability estimation if not yet run.
        Stores result as self.person_stats_{model}.

        Parameters
        ----------
        model : str, default 'global'
            Rater parameterisation.
        anchor_raters : list or None, default None
            If provided, uses anchor-calibrated abilities.
        full : bool, default False
            If True, sets rsem=True, zstd=True.
        rsem : bool, default False
            If True, includes Residual SEM (RSEM) column.
        zstd : bool, default False
            If True, includes Infit Z and Outfit Z columns.
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
        matrix_power : int, default 3
            Matrix power.
        log_lik_tol : float, default 0.000001
            Calibration convergence tolerance.
        interval : float or None, default None
            CI width (unused directly; passed to _ensure_calibrated).
        no_of_samples : int, default 100
            Bootstrap samples.

        Attributes set
        --------------
        person_stats_{model} : pandas.DataFrame
            Person statistics with persons as rows. Contains Estimate, CSEM,
            Score, Max score, p, Infit MS, Outfit MS. Optional: RSEM, Infit Z,
            Outfit Z.
        """

        self._ensure_calibrated(model, warm_corr=warm_corr, tolerance=tolerance,
                                max_iters=max_iters, ext_score_adjustment=ext_score_adjustment,
                                constant=constant, method=method, matrix_power=matrix_power,
                                log_lik_tol=log_lik_tol)
        if not hasattr(self, f'person_outfit_ms_{model}'):
            self._run_person_fit(model)
        if full:
            rsem = zstd = True

        anc  = anchor_raters is not None
        abils = (getattr(self, f'anchor_abils_{model}') if anc
                 else getattr(self, f'abils_{model}'))

        stats = pd.DataFrame(index=self.persons)
        stats['Estimate'] = abils.round(dp)
        stats['CSEM']     = getattr(self, f'csem_vector_{model}').round(dp)
        if rsem:
            stats['RSEM'] = getattr(self, f'rsem_vector_{model}').round(dp)

        unstacked         = self.dataframe.unstack(level=0)
        stats['Score']    = unstacked.sum(axis=1).astype(int)
        stats['Max score']= (unstacked.count(axis=1) * self.max_score).astype(int)
        stats['p']        = (unstacked.mean(axis=1) / self.max_score).round(dp)

        for col, src in [
            ('Infit MS',  getattr(self, f'person_infit_ms_{model}')),
            ('Outfit MS', getattr(self, f'person_outfit_ms_{model}')),
        ]:
            stats[col] = np.nan
            stats.loc[src.index, col] = src.round(dp).values
        if zstd:
            for col, src in [
                ('Infit Z',  getattr(self, f'person_infit_zstd_{model}')),
                ('Outfit Z', getattr(self, f'person_outfit_zstd_{model}')),
            ]:
                stats[col] = np.nan
                stats.loc[src.index, col] = src.round(dp).values

        setattr(self, f'person_stats_{model}', stats)

    def person_stats_df_global(self, **kw):     self.person_stats_df(model='global', **kw)
    def person_stats_df_items(self, **kw):      self.person_stats_df(model='items', **kw)
    def person_stats_df_thresholds(self, **kw): self.person_stats_df(model='thresholds', **kw)
    def person_stats_df_matrix(self, **kw):     self.person_stats_df(model='matrix', **kw)

    def test_stats_df(self, model='global', dp=3, warm_corr=True,
                      tolerance=0.00001, max_iters=100,
                      ext_score_adjustment=0.5, method='cos',
                      constant=0.1, matrix_power=3, log_lik_tol=0.000001,
                      no_of_samples=100):
        """
        Build and store the test-level summary statistics table.

        Auto-triggers calibration and test fit statistics if not yet run.
        Stores result as self.test_stats_{model}.

        Parameters
        ----------
        model : str, default 'global'
            Rater parameterisation.
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
        matrix_power : int, default 3
            Matrix power.
        log_lik_tol : float, default 0.000001
            Calibration convergence tolerance.
        no_of_samples : int, default 100
            Bootstrap samples.

        Attributes set
        --------------
        test_stats_{model} : pandas.DataFrame
            Two-column table (Items, Persons) with rows:
            Mean, SD, Separation ratio, Strata, Reliability.
        """

        self._ensure_calibrated(model, constant=constant, method=method,
                                matrix_power=matrix_power, log_lik_tol=log_lik_tol)
        if not hasattr(self, f'psi_{model}'):
            self._run_test_fit(model)

        stats = pd.DataFrame({
            'Items':   [self.diffs.mean(), self.diffs.std(),
                        getattr(self, f'isi_{model}'),
                        getattr(self, f'item_strata_{model}'),
                        getattr(self, f'item_reliability_{model}')],
            'Persons': [getattr(self, f'abils_{model}').mean(),
                        getattr(self, f'abils_{model}').std(),
                        getattr(self, f'psi_{model}'),
                        getattr(self, f'person_strata_{model}'),
                        getattr(self, f'person_reliability_{model}')],
        }, index=['Mean', 'SD', 'Separation ratio', 'Strata', 'Reliability'])
        setattr(self, f'test_stats_{model}', stats.round(dp))

    def test_stats_df_global(self, **kw):     self.test_stats_df(model='global', **kw)
    def test_stats_df_items(self, **kw):      self.test_stats_df(model='items', **kw)
    def test_stats_df_thresholds(self, **kw): self.test_stats_df(model='thresholds', **kw)
    def test_stats_df_matrix(self, **kw):     self.test_stats_df(model='matrix', **kw)

    # ------------------------------------------------------------------
    # Rater stats table (most complex -- varies substantially by model)
    # ------------------------------------------------------------------

    def rater_stats_df(self, model='global', anchor_raters=None,
                       full=False, zstd=False, marginal=True, dp=3,
                       warm_corr=True, tolerance=0.00001, max_iters=100,
                       ext_score_adjustment=0.5, method='cos', constant=0.1,
                       matrix_power=3, log_lik_tol=0.000001,
                       no_of_samples=100, interval=None):
        """
        Build and store the rater statistics summary table.

        Output structure varies substantially by model:
          global     — one row per rater with scalar severity estimate and fit stats.
          items      — MultiIndex columns (item, statistic), one row per rater.
          thresholds — MultiIndex columns (threshold, statistic), one row per rater.
          matrix     — marginal=True: twin-vector (per-item + per-threshold marginals
                       recentred to zero); marginal=False: full (item, threshold)
                       cell table.

        Auto-triggers the full calibration/SE/fit chain if not yet run.
        Stores result as self.rater_stats_{model}.

        Parameters
        ----------
        model : str, default 'global'
            Rater parameterisation.
        anchor_raters : list or None, default None
            Anchor raters for SE computation.
        full : bool, default False
            If True, sets zstd=True, interval=0.95.
        zstd : bool, default False
            If True, includes Infit Z and Outfit Z columns.
        marginal : bool, default True
            For matrix model only: if True returns marginal twin-vector
            representation; if False returns full (item, threshold) cell table.
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
        matrix_power : int, default 3
            Matrix power.
        log_lik_tol : float, default 0.000001
            Calibration convergence tolerance.
        no_of_samples : int, default 100
            Bootstrap samples.
        interval : float or None, default None
            CI width.

        Attributes set
        --------------
        rater_stats_{model} : pandas.DataFrame
            Rater statistics table. Structure depends on model (see above).
        """

        if full:
            zstd = True
            interval = interval or 0.95

        self._ensure_calibrated(model, anchor_raters=anchor_raters, interval=interval,
                                no_of_samples=no_of_samples, constant=constant,
                                method=method, matrix_power=matrix_power,
                                log_lik_tol=log_lik_tol, warm_corr=warm_corr,
                                tolerance=tolerance, max_iters=max_iters,
                                ext_score_adjustment=ext_score_adjustment)
        self._ensure_se(model, anchor_raters, interval, no_of_samples,
                        constant, method, matrix_power, log_lik_tol)
        if not hasattr(self, f'rater_outfit_ms_{model}'):
            self._run_rater_fit(model)

        anc = anchor_raters is not None
        rse = getattr(self, f'rater_se_{model}', {})
        rlo = getattr(self, f'rater_low_{model}', None)
        rhi = getattr(self, f'rater_high_{model}', None)

        if model == 'global':
            sev_attr = f'anchor_severities_{model}' if anc else f'severities_{model}'
            severities = getattr(self, sev_attr)
            stats = pd.DataFrame({'Estimate': severities.round(dp)})
            if rse is not None:
                stats['SE'] = pd.Series(rse).round(dp)
            if interval is not None and rlo is not None:
                stats[f'{round((1-interval)*50,1)}%'] = pd.Series(rlo).round(dp)
                stats[f'{round((1+interval)*50,1)}%'] = pd.Series(rhi).round(dp)
            stats['Count']    = pd.Series({r: self.dataframe.xs(r).count().sum()
                                           for r in self.raters})
            stats['Infit MS'] = getattr(self, f'rater_infit_ms_{model}').round(dp)
            if zstd:
                stats['Infit Z'] = getattr(self, f'rater_infit_zstd_{model}').round(dp)
            stats['Outfit MS'] = getattr(self, f'rater_outfit_ms_{model}').round(dp)
            if zstd:
                stats['Outfit Z'] = getattr(self, f'rater_outfit_zstd_{model}').round(dp)
            stats.index = self.raters
            setattr(self, f'rater_stats_{model}', stats)

        else:
            sev_attr = f'anchor_severities_{model}' if anc else f'severities_{model}'
            severities = getattr(self, sev_attr)
            se_attr  = f'anchor_rater_se_{model}'  if anc else f'rater_se_{model}'
            rse = getattr(self, se_attr, {})
            lo_attr = f'anchor_rater_low_{model}'  if anc else f'rater_low_{model}'
            hi_attr = f'anchor_rater_high_{model}' if anc else f'rater_high_{model}'
            rlo = getattr(self, lo_attr, None)
            rhi = getattr(self, hi_attr, None)

            def _ov_stats():
                cols = (['Count', 'Infit MS', 'Infit Z', 'Outfit MS', 'Outfit Z']
                        if zstd else ['Count', 'Infit MS', 'Outfit MS'])
                ov = pd.DataFrame(index=self.raters, columns=cols)
                ov['Count'] = pd.Series({r: self.dataframe.xs(r).count().sum()
                                         for r in self.raters}).astype(int)
                ov['Infit MS']  = getattr(self, f'rater_infit_ms_{model}').round(dp)
                ov['Outfit MS'] = getattr(self, f'rater_outfit_ms_{model}').round(dp)
                if zstd:
                    ov['Infit Z']  = getattr(self, f'rater_infit_zstd_{model}').round(dp)
                    ov['Outfit Z'] = getattr(self, f'rater_outfit_zstd_{model}').round(dp)
                return ov.T

            result = {}

            if model == 'items':
                for item in self.items:
                    sub = pd.DataFrame(index=self.raters)
                    sub['Estimate'] = np.array([severities[rater][item]
                                                for rater in self.raters]).round(dp)
                    sub['SE'] = np.array([rse.get(rater, {}).get(item, np.nan)
                                          for rater in self.raters]).round(dp)
                    if interval is not None and rlo is not None:
                        sub[f'{round((1-interval)*50,1)}%'] = np.array(
                            [rlo[rater][item] for rater in self.raters]).round(dp)
                        sub[f'{round((1+interval)*50,1)}%'] = np.array(
                            [rhi[rater][item] for rater in self.raters]).round(dp)
                    result[item] = sub.T

            elif model == 'thresholds':
                for t in range(self.max_score):
                    key = f'Threshold {t+1}'
                    sub = pd.DataFrame(index=self.raters)
                    sub['Estimate'] = np.array([severities[rater][t+1]
                                                for rater in self.raters]).round(dp)
                    sub['SE'] = np.array([rse.get(rater, np.zeros(self.max_score+1))[t+1]
                                          for rater in self.raters]).round(dp)
                    if interval is not None and rlo is not None:
                        sub[f'{round((1-interval)*50,1)}%'] = np.array(
                            [rlo[rater][t+1] for rater in self.raters]).round(dp)
                        sub[f'{round((1+interval)*50,1)}%'] = np.array(
                            [rhi[rater][t+1] for rater in self.raters]).round(dp)
                    result[key] = sub.T

            elif model == 'matrix':
                if marginal:
                    mg_i_attr = f'anchor_marginal_severities_items' if anc else 'marginal_severities_items'
                    mg_t_attr = f'anchor_marginal_severities_thresholds' if anc else 'marginal_severities_thresholds'
                    mg_items = getattr(self, mg_i_attr)
                    mg_thrs  = getattr(self, mg_t_attr)
                    mg_se_i  = getattr(self, 'rater_se_marginal_items', {})
                    mg_se_t  = getattr(self, 'rater_se_marginal_thresholds', {})

                    for item in self.items:
                        sub = pd.DataFrame(index=self.raters)
                        sub['Estimate'] = np.array([mg_items[rater][item]
                                                    for rater in self.raters]).round(dp)
                        sub['SE'] = np.array([mg_se_i.get(rater, {}).get(item, np.nan)
                                              for rater in self.raters]).round(dp)
                        result[item] = sub.T

                    for t in range(self.max_score):
                        key = f'Threshold {t+1}'
                        sub = pd.DataFrame(index=self.raters)
                        sub['Estimate'] = np.array([mg_thrs[rater][t+1]
                                                    for rater in self.raters]).round(dp)
                        sub['SE'] = np.array([
                            mg_se_t.get(rater, np.zeros(self.max_score+1))[t+1]
                            for rater in self.raters]).round(dp)
                        result[key] = sub.T

                else:
                    for item in self.items:
                        for t in range(self.max_score):
                            key = f'{item}, Threshold {t+1}'
                            sub = pd.DataFrame(index=self.raters)
                            sub['Estimate'] = np.array([severities[rater][item][t+1]
                                                        for rater in self.raters]).round(dp)
                            sub['SE'] = np.array([rse.get(rater, {}).get(
                                item, np.zeros(self.max_score+1))[t+1]
                                for rater in self.raters]).round(dp)
                            if interval is not None and rlo is not None:
                                sub[f'{round((1-interval)*50,1)}%'] = np.array(
                                    [rlo[rater][item][t+1] for rater in self.raters]).round(dp)
                                sub[f'{round((1+interval)*50,1)}%'] = np.array(
                                    [rhi[rater][item][t+1] for rater in self.raters]).round(dp)
                            result[key] = sub.T

            result['Overall statistics'] = _ov_stats()
            stats = pd.concat(result.values(), keys=result.keys()).T
            setattr(self, f'rater_stats_{model}', stats)

    def rater_stats_df_global(self, **kw):     self.rater_stats_df(model='global', **kw)
    def rater_stats_df_items(self, **kw):      self.rater_stats_df(model='items', **kw)
    def rater_stats_df_thresholds(self, **kw): self.rater_stats_df(model='thresholds', **kw)
    def rater_stats_df_matrix(self, **kw):     self.rater_stats_df(model='matrix', **kw)

    # ------------------------------------------------------------------
    # Save statistics
    # ------------------------------------------------------------------

    def save_stats(self, model='global', filename='', format='csv', dp=3,
                   warm_corr=True, tolerance=0.00001, max_iters=100,
                   ext_score_adjustment=0.5, method='cos', constant=0.1,
                   matrix_power=3, log_lik_tol=0.000001,
                   no_of_samples=100, interval=None):
        """
        Export item, threshold, rater, person, and test statistics to file.

        Auto-triggers all stats_df methods if not yet run. Saves all five
        tables to either a single Excel workbook or separate CSV files.

        Parameters
        ----------
        model : str, default 'global'
            Rater parameterisation.
        filename : str, default ''
            Output filename or path (without extension for CSV).
        format : str, default 'csv'
            'csv' saves five separate CSV files. 'xlsx' saves to a single
            workbook with separate sheets.
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
        matrix_power : int, default 3
            Matrix power.
        log_lik_tol : float, default 0.000001
            Calibration convergence tolerance.
        no_of_samples : int, default 100
            Bootstrap samples.
        interval : float or None, default None
            CI width for SEs.
        """

        kw = dict(dp=dp, warm_corr=warm_corr, tolerance=tolerance,
                  max_iters=max_iters, ext_score_adjustment=ext_score_adjustment,
                  method=method, constant=constant, matrix_power=matrix_power,
                  log_lik_tol=log_lik_tol)

        for attr, method_name, extra in [
            (f'item_stats_{model}',      'item_stats_df',      dict(no_of_samples=no_of_samples, interval=interval)),
            (f'threshold_stats_{model}', 'threshold_stats_df', dict(no_of_samples=no_of_samples, interval=interval)),
            (f'rater_stats_{model}',     'rater_stats_df',     dict(no_of_samples=no_of_samples, interval=interval)),
            (f'person_stats_{model}',    'person_stats_df',    {}),
            (f'test_stats_{model}',      'test_stats_df',      {}),
        ]:
            if not hasattr(self, attr):
                getattr(self, method_name)(model=model, **kw, **extra)

        if format == 'xlsx':
            if not filename.endswith('.xlsx'):
                filename += '.xlsx'
            with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
                getattr(self, f'item_stats_{model}').to_excel(
                    writer, sheet_name='Item statistics')
                getattr(self, f'threshold_stats_{model}').to_excel(
                    writer, sheet_name='Threshold statistics')
                getattr(self, f'rater_stats_{model}').to_excel(
                    writer, sheet_name='Rater statistics')
                getattr(self, f'person_stats_{model}').to_excel(
                    writer, sheet_name='Person statistics')
                getattr(self, f'test_stats_{model}').to_excel(
                    writer, sheet_name='Test statistics')
        else:
            if filename.endswith('.csv'):
                filename = filename[:-4]
            getattr(self, f'item_stats_{model}').to_csv(f'{filename}_item_stats.csv')
            getattr(self, f'threshold_stats_{model}').to_csv(f'{filename}_threshold_stats.csv')
            getattr(self, f'rater_stats_{model}').to_csv(f'{filename}_rater_stats.csv')
            getattr(self, f'person_stats_{model}').to_csv(f'{filename}_person_stats.csv')
            getattr(self, f'test_stats_{model}').to_csv(f'{filename}_test_stats.csv')

    def save_stats_global(self, **kw):     self.save_stats(model='global', **kw)
    def save_stats_items(self, **kw):      self.save_stats(model='items', **kw)
    def save_stats_thresholds(self, **kw): self.save_stats(model='thresholds', **kw)
    def save_stats_matrix(self, **kw):     self.save_stats(model='matrix', **kw)

    def save_residuals(self, eigenvectors, eigenvalues, variance_explained,
                       loadings, fit_statistics_method, eigenvector_string,
                       filename, format='csv', single=True, dp=3, **kw):
        """
        Export residual correlation analysis results to file.

        Low-level method called by save_residuals_items_* and
        save_residuals_raters_* aliases. Auto-triggers the fit statistics
        method if the eigenvectors attribute is not yet set.

        Parameters
        ----------
        eigenvectors : pandas.DataFrame or None
            PCA eigenvectors to save.
        eigenvalues : pandas.DataFrame or None
            PCA eigenvalues to save.
        variance_explained : pandas.DataFrame or None
            PCA variance explained proportions to save.
        loadings : pandas.DataFrame or None
            PCA loadings to save.
        fit_statistics_method : str
            Name of the method to call if eigenvectors are not yet computed
            (e.g. 'item_res_corr_analysis_global').
        eigenvector_string : str
            Attribute name to check for existence (e.g. 'item_eigenvectors_global').
        filename : str
            Output filename or path.
        format : str, default 'csv'
            'csv' or 'xlsx'.
        single : bool, default True
            If True, writes all tables to a single file/sheet.
            If False, writes each to a separate file/sheet.
        dp : int, default 3
            Decimal places.
        **kw
            Additional keyword arguments passed to the fit statistics method.
        """

        frames  = [eigenvectors, eigenvalues, variance_explained, loadings]
        if not hasattr(self, eigenvector_string):
            getattr(self, fit_statistics_method)(**kw)

        if format == 'xlsx':
            if not filename.endswith('.xlsx'):
                filename += '.xlsx'
            with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
                if single:
                    row = 0
                    for frame in frames:
                        frame.round(dp).to_excel(writer,
                                                  sheet_name='Residual analysis',
                                                  startrow=row, startcol=0)
                        row += frame.shape[0] + 2
                else:
                    for frame, sheet in zip(frames, ['Eigenvectors', 'Eigenvalues',
                                                      'Variance explained',
                                                      'Loadings']):
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
                for frame, suffix in zip(frames, ['_eigenvectors', '_eigenvalues',
                                                    '_variance_explained',
                                                    '_loadings']):
                    frame.round(dp).to_csv(f'{filename}{suffix}.csv')

    def _save_residuals_for(self, model, which, filename, **kw):
        '''Shared implementation for save_residuals_items/raters aliases.'''
        attr = f'{which}_eigenvectors_{model}'
        if not hasattr(self, attr):
            runner = self._run_item_res_corr if which == 'item' else self._run_rater_res_corr
            runner(model, **kw)
        self.save_residuals(
            getattr(self, f'{which}_eigenvectors_{model}'),
            getattr(self, f'{which}_eigenvalues_{model}'),
            getattr(self, f'{which}_variance_explained_{model}'),
            getattr(self, f'{which}_loadings_{model}'),
            f'{which}_res_corr_analysis_{model}',
            attr, filename, **kw
        )

    def save_residuals_items_global(self, filename, **kw):
        self._save_residuals_for('global', 'item', filename, **kw)
    def save_residuals_items_items(self, filename, **kw):
        self._save_residuals_for('items', 'item', filename, **kw)
    def save_residuals_items_thresholds(self, filename, **kw):
        self._save_residuals_for('thresholds', 'item', filename, **kw)
    def save_residuals_items_matrix(self, filename, **kw):
        self._save_residuals_for('matrix', 'item', filename, **kw)

    def save_residuals_raters_global(self, filename, **kw):
        self._save_residuals_for('global', 'rater', filename, **kw)
    def save_residuals_raters_items(self, filename, **kw):
        self._save_residuals_for('items', 'rater', filename, **kw)
    def save_residuals_raters_thresholds(self, filename, **kw):
        self._save_residuals_for('thresholds', 'rater', filename, **kw)
    def save_residuals_raters_matrix(self, filename, **kw):
        self._save_residuals_for('matrix', 'rater', filename, **kw)


    # ------------------------------------------------------------------
    # Class intervals
    # ------------------------------------------------------------------

    @staticmethod
    def _class_masks(abils, no_of_classes):
        '''Compute class interval index masks from ability values.'''
        class_groups = [f'class_{i + 1}' for i in range(no_of_classes)]
        q = abils.quantile([(i + 1) / no_of_classes
                            for i in range(no_of_classes - 1)])
        mask = {
            'class_1':              abils < q.values[0],
            f'class_{no_of_classes}': abils >= q.values[-1],
            **{f'class_{i + 2}':  ((abils >= q.values[i]) &
                                    (abils < q.values[i + 1]))
               for i in range(no_of_classes - 2)}
        }
        return {cg: mask[cg][mask[cg]].index for cg in class_groups}

    def _severity_item_offset(self, model, severities, rater):
        '''Return per-item severity offset Series for a given rater and model.'''
        if model == 'global':
            return pd.Series(float(severities.loc[rater].iloc[0]) if isinstance(severities.loc[rater], pd.Series) else float(severities.loc[rater]), index=self.items)
        elif model == 'items':
            return pd.Series({item: severities[rater][item] for item in self.items})
        elif model == 'thresholds':
            return pd.Series(float(severities[rater][1:].mean()), index=self.items)
        elif model == 'matrix':
            return pd.Series({item: np.mean(severities[rater][item][1:])
                              for item in self.items})

    def class_intervals(self, abilities, items=None, raters=None,
                        shift=0, no_of_classes=5):
        '''Class intervals for TCC/ICC observed data overlay.'''
        if isinstance(items, str) and items in ('all', 'none'):
            items = None
        if isinstance(raters, str):
            if raters in ('none', 'zero'):
                raters = None
            elif raters == 'all':
                raters = self.raters.tolist()
            else:
                raters = [raters]

        class_groups = [f'class_{i + 1}' for i in range(no_of_classes)]
        df = self.dataframe.copy()

        # Get person index (persons with non-missing data on relevant items)
        if items is None:
            abil_index = self.dataframe.unstack(level=0).dropna(how='any').index
        else:
            abil_index = self.dataframe[items].unstack(level=0).dropna(how='any').index

        abils = abilities.loc[abil_index]

        # Subset by raters
        if isinstance(raters, list):
            df = pd.concat({r: df.xs(r) for r in raters}, keys=raters)

        # Subset by items (after rater subsetting to preserve index structure)
        if items is not None:
            df = df[items]

		# Subset by person index — handle string vs list items separately
        # When items is a single string, df[items] is a Series; pd.IndexSlice
        # with three levels raises "Too many indexers" on a Series, so use
        # xs+loc instead.
        if isinstance(items, str):
            rater_list = raters if isinstance(raters, list) else list(self.raters)
            df = pd.concat({r: df.xs(r).loc[abil_index] for r in rater_list},
            				keys=rater_list)
        elif isinstance(items, list):
            df = df.loc[pd.IndexSlice[:, abil_index], :]
        else:
            df = df.loc[pd.IndexSlice[:, abil_index], :]

        # Class quantile masks
        quantiles = abils.quantile([(i + 1) / no_of_classes
                                    for i in range(no_of_classes - 1)])
        mask_dict = {'class_1': abils < quantiles.values[0],
                     f'class_{no_of_classes}': abils >= quantiles.values[-1]}
        for i in range(no_of_classes - 2):
            mask_dict[f'class_{i + 2}'] = ((abils >= quantiles.values[i]) &
                                            (abils < quantiles.values[i + 1]))

        # Expand masks to (Rater, Person) MultiIndex
        rater_list = list(self.raters) if raters is None else raters
        df_mask_dict = {}
        for cg in class_groups:
            expanded = pd.concat({r: mask_dict[cg] for r in rater_list},
                                 keys=rater_list)
            df_mask_dict[cg] = expanded[expanded].index

        mean_abilities = pd.Series({cg: abils[mask_dict[cg]].mean()
                                    for cg in class_groups}) - shift

        if raters is None:
            obs = pd.Series({cg: df.loc[df_mask_dict[cg]].mean().sum()
                             for cg in class_groups})
        else:
            obs = pd.Series({
                cg: sum(
                    df.xs(r).loc[
                        df_mask_dict[cg][
                            df_mask_dict[cg].get_level_values(0) == r
                        ].get_level_values(1)
                    ].mean().sum()
                    if (df_mask_dict[cg].get_level_values(0) == r).any()
                    else 0.0
                    for r in raters
                )
                for cg in class_groups
            })

        return mean_abilities, obs

    def class_intervals_cats(self, abilities, difficulties, thresholds,
                              severities, model='global', item=None,
                              rater=None, shift=0, no_of_classes=5):
        '''Class intervals for CRC observed data overlay.'''
        if rater in ('none', 'zero'):
            rater = None

        class_groups = [f'class_{i + 1}' for i in range(no_of_classes)]
        df = self.dataframe.copy()

        # Build ability DataFrame: (Person, Items)
        abil_df = pd.DataFrame({it: abilities for it in self.dataframe.columns})
        if item is None:
            for it in self.dataframe.columns:
                abil_df[it] -= float(difficulties[it])

        # Subtract rater severity from ability
        abil_dict = {}
        for r in self.raters:
            a = abil_df.copy()
            if rater is None:
                sev = self._severity_item_offset(model, severities, r)
                for it in self.dataframe.columns:
                    a[it] -= float(sev[it])
            abil_dict[r] = a
        abil_df_full = pd.concat(abil_dict.values(), keys=abil_dict.keys())

        # Subset by item/rater
        if item is None and rater is None:
            pf = self.dataframe.notna().astype(float).replace(0, np.nan)
            abil_full = abil_df_full * pf
            mask_scores = df.unstack().unstack()
            mask_abils  = abil_full.unstack().unstack()
        elif item is None and rater is not None:
            df_r = df.xs(rater)
            pf   = df_r.notna().astype(float).replace(0, np.nan)
            mask_scores = df_r.unstack()
            mask_abils  = (abil_df_full.xs(rater) * pf).unstack()
        elif item is not None and rater is None:
            df_i = df[item].unstack(level=0)
            pf   = df_i.notna().astype(float).replace(0, np.nan)
            mask_scores = df_i.unstack()
            mask_abils  = (abil_df_full[item].unstack(level=0) * pf).unstack()
        else:
            df_ri = df.xs(rater)[item]
            pf    = df_ri.notna().astype(float).replace(0, np.nan)
            mask_scores = df_ri
            mask_abils  = abil_df_full.xs(rater)[item] * pf

        masks = self._class_masks(mask_abils, no_of_classes)
        mean_abilities = np.array([
            mask_abils.loc[masks[cg]].mean() for cg in class_groups
        ])
        obs_props = np.array([
            [(mask_scores.loc[masks[cg]] == cat).sum() / len(masks[cg])
             for cg in class_groups]
            for cat in range(self.max_score + 1)
        ])
        return mean_abilities, obs_props

    def class_intervals_thr(self, abilities, difficulties, severities,
                             model='global', item=None, rater=None,
                             shift=None, no_of_classes=5):
        '''Class intervals for threshold CCC observed data overlay.'''
        if item in ('none',):
            item = None
        if rater in ('none', 'zero'):
            rater = None
        if shift is None:
            shift = 0

        class_groups = [f'class_{i + 1}' for i in range(no_of_classes)]
        df = self.dataframe.copy()

        abil_df = pd.DataFrame({it: abilities for it in self.dataframe.columns})
        if item is None:
            for it in self.dataframe.columns:
                abil_df[it] -= float(difficulties[it])

        abil_dict = {}
        for r in self.raters:
            a = abil_df.copy()
            if rater is None:
                sev = self._severity_item_offset(model, severities, r)
                for it in self.dataframe.columns:
                    a[it] -= float(sev[it])
            abil_dict[r] = a
        abil_df_full = pd.concat(abil_dict.values(), keys=abil_dict.keys())

        if item is not None:
            df = df[item]
            abil_df_full = abil_df_full[item]
        if rater is not None:
            df = df.xs(rater)
            abil_df_full = abil_df_full.xs(rater)

        mean_abilities_all, obs_props_all = [], []
        for t in range(self.max_score):
            cond_df   = df[df.isin([t, t + 1])] - t
            cond_mask = cond_df.notna().astype(float).replace(0, np.nan)
            cond_abils = abil_df_full * cond_mask

            if item is None:
                obs_data = pd.DataFrame({
                    'ability': cond_abils.stack(),
                    'score':   cond_df.stack()
                }).droplevel(level=1)
            else:
                obs_data = pd.DataFrame({'ability': cond_abils, 'score': cond_df})

            masks = self._class_masks(obs_data['ability'], no_of_classes)
            mean_abilities_all.append([
                obs_data.loc[masks[cg]]['ability'].mean() + shift
                for cg in class_groups
            ])
            obs_props_all.append([
                obs_data.loc[masks[cg]]['score'].mean()
                for cg in class_groups
            ])

        return np.array(mean_abilities_all), np.array(obs_props_all)

    # Backwards-compatible per-model aliases
    def class_intervals_cats_global(self, abilities, difficulties, thresholds, severities, **kw):
        return self.class_intervals_cats(abilities, difficulties, thresholds, severities, 'global', **kw)
    def class_intervals_cats_items(self, abilities, difficulties, thresholds, severities, **kw):
        return self.class_intervals_cats(abilities, difficulties, thresholds, severities, 'items', **kw)
    def class_intervals_cats_thresholds(self, abilities, difficulties, thresholds, severities, **kw):
        return self.class_intervals_cats(abilities, difficulties, thresholds, severities, 'thresholds', **kw)
    def class_intervals_cats_matrix(self, abilities, difficulties, thresholds, severities, **kw):
        return self.class_intervals_cats(abilities, difficulties, thresholds, severities, 'matrix', **kw)

    def class_intervals_thr_global(self, abilities, difficulties, severities, **kw):
        return self.class_intervals_thr(abilities, difficulties, severities, 'global', **kw)
    def class_intervals_thr_items(self, abilities, difficulties, severities, **kw):
        return self.class_intervals_thr(abilities, difficulties, severities, 'items', **kw)
    def class_intervals_thr_thresholds(self, abilities, difficulties, severities, **kw):
        return self.class_intervals_thr(abilities, difficulties, severities, 'thresholds', **kw)
    def class_intervals_thr_matrix(self, abilities, difficulties, severities, **kw):
        return self.class_intervals_thr(abilities, difficulties, severities, 'matrix', **kw)


    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    def plot_data(self,
                  x_data,
                  y_data,
                  model='global',
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
        '''
        Core plotting function for ability-function curves (MFRM).
        Shared across all four rater parameterisations.
        '''
        difficulties, thresholds, severities = self._get_params(model, anchor)

        if isinstance(raters, str):
            raters = (None if raters in ('none', 'zero', 'all')
                      else [raters])
        if isinstance(items, str):
            items = None if items == 'all' else items

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
            color_map = (sns.color_palette('dark', as_cmap=True) if palette == 'dark multi'
                         else sns.dark_palette(base_color, reverse=True, as_cmap=True))
        else:
            color_map = (sns.color_palette('muted', as_cmap=True) if palette == 'light multi'
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
                    if isinstance(y_obs_data, pd.Series):
                        col = scalarMap.to_rgba(0) if 'multi' not in palette else color_map[0]
                        ax.plot(x_obs_data, y_obs_data, 'o', color=col)
                    else:
                        for j in range(y_obs_data.shape[0]):
                            col = (scalarMap.to_rgba(j) if 'multi' not in palette
                                   else color_map[j])
                            ax.plot(x_obs_data, y_obs_data[j, :], 'o', color=col)
                except Exception:
                    pass

            if thresh_obs is not None:
                try:
                    for j in range(x_obs_data.shape[0]):
                        col = (scalarMap.to_rgba(j) if 'multi' not in palette
                               else color_map[j])
                        ax.plot(x_obs_data[j, :], y_obs_data[j, :],
                                'o', color=col)
                except Exception:
                    pass

            if thresh_lines:
                for t in range(self.max_score):
                    if items is None and raters is None:
                        xval = thresholds[t + 1]
                    elif items is None:
                        r_sc = raters[0] if isinstance(raters, list) else raters
                        xval = thresholds[t + 1] + float(
                            self._severity_item_offset(model, severities, r_sc).mean()
                        )
                    else:
                        xval = float(difficulties[items]) + thresholds[t + 1]
                    ax.axvline(x=xval, color='black', linestyle='--')

            if central_diff:
                if items is None:
                    ax.axvline(x=0, color='darkred', linestyle='--')
                else:
                    ax.axvline(x=float(difficulties[items]), color='darkred', linestyle='--')

            if score_lines_item[1] is not None:
                item = score_lines_item[0]
                if (all(s > 0 for s in score_lines_item[1]) and
                        all(s < self.max_score for s in score_lines_item[1])):
                    for s in score_lines_item[1]:
                        abil = self.score_abil(s, model=model, anchor=anchor,
                                               items=[item] if item else None,
                                               raters=raters, warm_corr=False)
                        ax.vlines(x=abil, ymin=-100, ymax=s,
                                  color='black', linestyles='dashed')
                        ax.hlines(y=s, xmin=-100, xmax=abil,
                                  color='black', linestyles='dashed')
                        if score_labels:
                            ax.text(abil + (x_max - x_min) / 100, y_max / 50,
                                    str(round(abil, 2)))
                            ax.text(x_min + (x_max - x_min) / 100, s + y_max / 50, str(s))
                else:
                    warnings.warn('Invalid score for score line: values must be '
                                  'strictly between 0 and the item maximum score.',
                                  UserWarning, stacklevel=2)

            if score_lines_test is not None:
                item_keys = list(self.items) if items is None else (
                    [items] if isinstance(items, str) else items
                )
                n_items = len(item_keys)
                n_raters = len(raters) if raters is not None else self.no_of_raters
                max_total = self.max_score * n_items * n_raters
                if all(0 < s < max_total for s in score_lines_test):
                    for s in score_lines_test:
                        abil = self.score_abil(s, model=model, anchor=anchor,
                                               items=item_keys, raters=raters,
                                               warm_corr=warm)
                        ax.vlines(x=abil, ymin=-100, ymax=s,
                                  color='black', linestyles='dashed')
                        ax.hlines(y=s, xmin=-100, xmax=abil,
                                  color='black', linestyles='dashed')
                        if score_labels:
                            ax.text(abil + (x_max - x_min) / 100, y_max / 50,
                                    str(round(abil, 2)))
                            ax.text(x_min + (x_max - x_min) / 100, s + y_max / 50, str(s))
                else:
                    warnings.warn('Invalid score for score line: values must be '
                                  'strictly between 0 and the test maximum score.',
                                  UserWarning, stacklevel=2)

            if point_info_lines_item[1] is not None:
                item = point_info_lines_item[0]
                r    = (raters[0] if isinstance(raters, list) else
                    raters if raters is not None else list(self.raters)[0])
                for abil in point_info_lines_item[1]:
                    info = self.variance(abil, item, difficulties, r,
                                         severities, thresholds, model)
                    ax.vlines(x=abil, ymin=-100, ymax=info,
                              color='black', linestyles='dashed')
                    ax.hlines(y=info, xmin=-100, xmax=abil,
                              color='black', linestyles='dashed')
                    if score_labels:
                        ax.text(abil + (x_max - x_min) / 100, y_max / 50,
                                str(round(abil, 2)))
                        ax.text(x_min + (x_max - x_min) / 100, info + y_max / 50,
                                str(round(info, 3)))

            if point_info_lines_test is not None:
                item_keys  = list(self.items) if items is None else items
                rater_list = list(self.raters) if raters is None else raters
                for abil in point_info_lines_test:
                    info = sum(
                        self.variance(abil, it, difficulties, r,
                                      severities, thresholds, model)
                        for it in item_keys for r in rater_list
                    )
                    ax.vlines(x=abil, ymin=-100, ymax=info,
                              color='black', linestyles='dashed')
                    ax.hlines(y=info, xmin=-100, xmax=abil,
                              color='black', linestyles='dashed')
                    if score_labels:
                        ax.text(abil + (x_max - x_min) / 100, y_max / 50,
                                str(round(abil, 2)))
                        ax.text(x_min + (x_max - x_min) / 100, info + y_max / 50,
                                str(round(info, 3)))

            if point_csem_lines is not None:
                item_keys  = list(self.items) if items is None else items
                rater_list = list(self.raters) if raters is None else raters
                for abil in point_csem_lines:
                    info = sum(
                        self.variance(abil, it, difficulties, r,
                                      severities, thresholds, model)
                        for it in item_keys for r in rater_list
                    )
                    csem = 1.0 / (info ** 0.5)
                    ax.vlines(x=abil, ymin=-100, ymax=csem,
                              color='black', linestyles='dashed')
                    ax.hlines(y=csem, xmin=-100, xmax=abil,
                              color='black', linestyles='dashed')
                    if score_labels:
                        ax.text(abil + (x_max - x_min) / 100, y_max / 50,
                                str(round(abil, 2)))
                        ax.text(x_min + (x_max - x_min) / 100, csem + y_max / 50,
                                str(round(csem, 3)))

            if cat_highlight in range(self.max_score + 1):
                sev_shift = 0.0
                if raters is not None:
                    # _severity_item_offset expects a scalar rater
                    r_scalar = raters[0] if isinstance(raters, list) else raters
                    sev_shift = float(
                        self._severity_item_offset(model, severities, r_scalar).mean()
                    )
                diff_shift = 0.0 if items is None else float(difficulties[items])

                if cat_highlight == 0:
                    ax.axvspan(-100, diff_shift + thresholds[1] + sev_shift,
                               facecolor='blue', alpha=0.2)
                elif cat_highlight == self.max_score:
                    ax.axvspan(diff_shift + thresholds[self.max_score] + sev_shift,
                               100, facecolor='blue', alpha=0.2)
                else:
                    lo = diff_shift + thresholds[cat_highlight] + sev_shift
                    hi = diff_shift + thresholds[cat_highlight + 1] + sev_shift
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

    # Backwards-compatible plot_data aliases
    def plot_data_global(self, *args, **kw):
        return self.plot_data(*args, model='global', **kw)
    def plot_data_items(self, *args, **kw):
        return self.plot_data(*args, model='items', **kw)
    def plot_data_thresholds(self, *args, **kw):
        return self.plot_data(*args, model='thresholds', **kw)
    def plot_data_matrix(self, *args, **kw):
        return self.plot_data(*args, model='matrix', **kw)

    # ------------------------------------------------------------------
    # ICC, CRCS, Threshold CCS, IIC, TCC, Test info, Test CSEM, Residuals
    # ------------------------------------------------------------------

    def icc(self, item, model='global', anchor=False, rater=None, obs=None,
            warm=True, xmin=-5, xmax=5, no_of_classes=5, title=None,
            thresh_lines=False, score_lines=None, score_labels=False,
            central_diff=False, cat_highlight=None, plot_style='white',
            palette='dark blue', black=False, font='Times New Roman',
            title_font_size=15, axis_font_size=12, labelsize=12,
            filename=None, file_format='png', dpi=300):
        '''Item Characteristic Curve.'''
        difficulties, thresholds, severities = self._get_params(model, anchor)
        if rater in ('none', 'zero'):
            rater = None

        if obs:
            if not hasattr(self, f'abils_{model}'):
                self.person_abils(model=model)
            abilities = getattr(self, f'abils_{model}')
            xobsdata, yobsdata = self.class_intervals(
                abilities, items=item, raters=rater,
                no_of_classes=no_of_classes
            )
            yobsdata = np.array(yobsdata).reshape(-1, 1)
        else:
            xobsdata = yobsdata = np.array(np.nan)

        abilities = np.arange(-20, 20, 0.1)
        r_use = (rater if rater is not None
                 else list(self.raters)[0])
        y = np.array([
            self.exp_score(a, item, difficulties, r_use,
                           severities, thresholds, model)
            for a in abilities
        ]).reshape(-1, 1)

        return self.plot_data(
            x_data=abilities, y_data=y, model=model, anchor=anchor,
            items=item, raters=rater, obs=obs, warm=warm,
            x_obs_data=xobsdata, y_obs_data=yobsdata,
            x_min=xmin, x_max=xmax, y_max=self.max_score,
            thresh_lines=thresh_lines, graph_title=title or '',
            score_lines_item=[item, score_lines], score_labels=score_labels,
            central_diff=central_diff, cat_highlight=cat_highlight,
            y_label='Expected score', plot_style=plot_style, palette=palette,
            black=black, font=font, title_font_size=title_font_size,
            axis_font_size=axis_font_size, labelsize=labelsize,
            filename=filename, plot_density=dpi, file_format=file_format
        )

    def icc_global(self, item, **kw):     return self.icc(item, model='global', **kw)
    def icc_items(self, item, **kw):      return self.icc(item, model='items', **kw)
    def icc_thresholds(self, item, **kw): return self.icc(item, model='thresholds', **kw)
    def icc_matrix(self, item, **kw):     return self.icc(item, model='matrix', **kw)

    def crcs(self, model='global', anchor=False, item=None, rater=None,
             obs=None, no_of_classes=5, title=None, thresh_lines=False,
             central_diff=False, cat_highlight=None, xmin=-5, xmax=5,
             plot_style='white', palette='dark blue', black=False,
             font='Times New Roman', title_font_size=15, axis_font_size=12,
             labelsize=12, filename=None, file_format='png', dpi=300):
        '''Category Response Curves.'''
        difficulties, thresholds, severities = self._get_params(model, anchor)
        if item in ('none',): item = None
        if rater in ('none', 'zero'): rater = None

        if obs is not None:
            if not hasattr(self, f'abils_{model}'):
                self.person_abils(model=model)
            abilities = getattr(self, f'abils_{model}')
            xobsdata, yobsdata = self.class_intervals_cats(
                abilities, difficulties, thresholds, severities,
                model=model, item=item, rater=rater, no_of_classes=no_of_classes
            )
            if isinstance(obs, str) and obs == 'all':
                obs = np.arange(self.max_score + 1)
            if not all(c in np.arange(self.max_score + 1) for c in obs):
                warnings.warn("Invalid 'obs' value. Valid values are None, 'all', "
                              'or a list of category indices.',
                              UserWarning, stacklevel=2)
                return
            yobsdata = yobsdata[obs, :]
        else:
            xobsdata = yobsdata = np.array(np.nan)

        abilities_arr = np.arange(-20, 20, 0.1)
        r_use = (rater[0] if isinstance(rater, list) else
                 rater if rater is not None else list(self.raters)[0])
        diff_use = 0.0 if item is None else float(difficulties[item])
        y = np.array([
            [self.cat_prob(a, (item or list(self.items)[0]),
                           difficulties, r_use, severities, cat, thresholds, model)
             for cat in range(self.max_score + 1)]
            for a in abilities_arr
        ])

        return self.plot_data(
            x_data=abilities_arr, y_data=y, model=model, anchor=anchor,
            items=item, raters=rater, obs=obs,
            x_obs_data=xobsdata, y_obs_data=yobsdata,
            x_min=xmin, x_max=xmax, y_max=1,
            thresh_lines=thresh_lines, central_diff=central_diff,
            cat_highlight=cat_highlight, graph_title=title or '',
            y_label='Probability', plot_style=plot_style, palette=palette,
            black=black, font=font, title_font_size=title_font_size,
            axis_font_size=axis_font_size, labelsize=labelsize,
            filename=filename, plot_density=dpi, file_format=file_format
        )

    def crcs_global(self, item=None, **kw):     return self.crcs(model='global', item=item, **kw)
    def crcs_items(self, item=None, **kw):      return self.crcs(model='items', item=item, **kw)
    def crcs_thresholds(self, item=None, **kw): return self.crcs(model='thresholds', item=item, **kw)
    def crcs_matrix(self, item=None, **kw):     return self.crcs(model='matrix', item=item, **kw)

    def threshold_ccs(self, model='global', anchor=False, item=None,
                      rater=None, obs=None, no_of_classes=5, title=None,
                      thresh_lines=False, central_diff=False, cat_highlight=None,
                      xmin=-5, xmax=5, plot_style='white', palette='dark blue',
                      black=False, font='Times New Roman', title_font_size=15,
                      axis_font_size=12, labelsize=12, filename=None,
                      file_format='png', dpi=300):
        '''Threshold Characteristic Curves.'''
        difficulties, thresholds, severities = self._get_params(model, anchor)
        if item in ('none',): item = None
        if rater in ('none', 'zero'): rater = None

        xobsdata = yobsdata = np.array(np.nan)
        if obs is not None:
            if not hasattr(self, f'abils_{model}'):
                self.person_abils(model=model)
            abilities = getattr(self, f'abils_{model}')
            mean_abs, obs_props = self.class_intervals_thr(
                abilities, difficulties, severities,
                model=model, item=item, rater=rater, no_of_classes=no_of_classes
            )
            xobsdata, yobsdata = mean_abs, obs_props
            if obs != 'all':
                if not all(c in np.arange(self.max_score) + 1 for c in obs):
                    warnings.warn("Invalid 'obs' value. Valid values are None, 'all', "
                                  'or a list of threshold numbers.',
                                  UserWarning, stacklevel=2)
                    return
                obs_idx  = [o - 1 for o in obs]
                xobsdata = xobsdata[obs_idx, :]
                yobsdata = yobsdata[obs_idx, :]

        abilities_arr = np.arange(-20, 20, 0.1)
        r_use = (rater[0] if isinstance(rater, list) else
                 rater if rater is not None else list(self.raters)[0])
        diff_shift = 0.0 if item is None else float(difficulties[item])

        # Threshold locations including rater severity offset
        sev_offset = float(self._severity_item_offset(model, severities, r_use).mean())
        abs_thresh = thresholds[1:] + diff_shift + sev_offset
        y = np.array([
            [1.0 / (1.0 + np.exp(thr - a)) for thr in abs_thresh]
            for a in abilities_arr
        ])

        return self.plot_data(
            x_data=abilities_arr, y_data=y, model=model, anchor=anchor,
            items=item, raters=rater, obs=None, thresh_obs=obs,
            x_obs_data=xobsdata, y_obs_data=yobsdata,
            x_min=xmin, x_max=xmax, y_max=1,
            thresh_lines=thresh_lines, central_diff=central_diff,
            cat_highlight=cat_highlight, graph_title=title or '',
            y_label='Probability', plot_style=plot_style, palette=palette,
            black=black, font=font, title_font_size=title_font_size,
            axis_font_size=axis_font_size, labelsize=labelsize,
            filename=filename, file_format=file_format, plot_density=dpi
        )

    def threshold_ccs_global(self, item=None, **kw):     return self.threshold_ccs(model='global', item=item, **kw)
    def threshold_ccs_items(self, item=None, **kw):      return self.threshold_ccs(model='items', item=item, **kw)
    def threshold_ccs_thresholds(self, item=None, **kw): return self.threshold_ccs(model='thresholds', item=item, **kw)
    def threshold_ccs_matrix(self, item=None, **kw):     return self.threshold_ccs(model='matrix', item=item, **kw)

    def iic(self, item, model='global', anchor=False, rater=None, ymax=None,
            thresh_lines=False, central_diff=False, point_info_lines=None,
            point_info_labels=False, cat_highlight=None, title=None,
            xmin=-5, xmax=5, plot_style='white', palette='dark blue',
            black=False, font='Times New Roman', title_font_size=15,
            axis_font_size=12, labelsize=12, filename=None,
            file_format='png', dpi=300):
        '''Item Information Curve.'''
        difficulties, thresholds, severities = self._get_params(model, anchor)
        r_use = (rater[0] if isinstance(rater, list) else
                 rater if rater is not None and rater not in ('none', 'zero')
                 else list(self.raters)[0])
        abilities = np.arange(-20, 20, 0.1)
        y = np.array([
            self.variance(a, item, difficulties, r_use,
                          severities, thresholds, model)
            for a in abilities
        ]).reshape(-1, 1)
        if ymax is None:
            ymax = float(y.max()) * 1.1
        return self.plot_data(
            x_data=abilities, y_data=y, model=model, anchor=anchor,
            items=item, raters=rater, x_min=xmin, x_max=xmax, y_max=ymax,
            thresh_lines=thresh_lines, central_diff=central_diff,
            point_info_lines_item=[item, point_info_lines],
            score_labels=point_info_labels, cat_highlight=cat_highlight,
            graph_title=title or '', y_label='Fisher information',
            plot_style=plot_style, palette=palette, black=black, font=font,
            title_font_size=title_font_size, axis_font_size=axis_font_size,
            labelsize=labelsize, filename=filename, plot_density=dpi,
            file_format=file_format
        )

    def iic_global(self, item, **kw):     return self.iic(item, model='global', **kw)
    def iic_items(self, item, **kw):      return self.iic(item, model='items', **kw)
    def iic_thresholds(self, item, **kw): return self.iic(item, model='thresholds', **kw)
    def iic_matrix(self, item, **kw):     return self.iic(item, model='matrix', **kw)

    def tcc(self, model='global', anchor=False, items=None, raters=None,
            obs=False, no_of_classes=5, title=None, score_lines=None,
            score_labels=False, xmin=-5, xmax=5, plot_style='white',
            palette='dark blue', black=False, font='Times New Roman',
            title_font_size=15, axis_font_size=12, labelsize=12,
            filename=None, file_format='png', dpi=300):
        '''Test Characteristic Curve.'''
        difficulties, thresholds, severities = self._get_params(model, anchor)
        if isinstance(items, str) and items in ('all', 'none'):
            items = None
        if isinstance(raters, str) and raters in ('all', 'none', 'zero'):
            raters = None

        xobsdata = yobsdata = np.array(np.nan)
        if obs:
            if not hasattr(self, f'abils_{model}'):
                self.person_abils(model=model)
            abilities = getattr(self, f'abils_{model}')
            mean_abs, obs_means = self.class_intervals(
                abilities, items=items, raters=raters,
                no_of_classes=no_of_classes
            )
            xobsdata = mean_abs
            yobsdata = np.array(obs_means).reshape(no_of_classes, 1)

        item_keys  = list(self.items) if items is None else (
            [items] if isinstance(items, str) else items
        )
        rater_list = list(self.raters) if raters is None else (
            [raters] if isinstance(raters, str) else raters
        )

        abilities_arr = np.arange(-20, 20, 0.1)
        y = np.array([
            sum(self.exp_score(a, it, difficulties, r,
                               severities, thresholds, model)
                for it in item_keys for r in rater_list)
            for a in abilities_arr
        ]).reshape(-1, 1)
        y_max = self.max_score * len(item_keys) * len(rater_list)

        return self.plot_data(
            x_data=abilities_arr, y_data=y, model=model, anchor=anchor,
            items=items, raters=raters, obs=obs,
            x_obs_data=xobsdata, y_obs_data=yobsdata,
            x_min=xmin, x_max=xmax, y_max=y_max,
            score_lines_test=score_lines, score_labels=score_labels,
            graph_title=title or '', y_label='Expected score',
            plot_style=plot_style, palette=palette, black=black, font=font,
            title_font_size=title_font_size, axis_font_size=axis_font_size,
            labelsize=labelsize, filename=filename, plot_density=dpi,
            file_format=file_format
        )

    def tcc_global(self, **kw):     return self.tcc(model='global', **kw)
    def tcc_items(self, **kw):      return self.tcc(model='items', **kw)
    def tcc_thresholds(self, **kw): return self.tcc(model='thresholds', **kw)
    def tcc_matrix(self, **kw):     return self.tcc(model='matrix', **kw)

    def test_info(self, model='global', anchor=False, items=None, raters=None,
                  point_info_lines=None, point_info_labels=False, xmin=-5,
                  xmax=5, ymax=None, title=None, plot_style='white',
                  palette='dark blue', black=False, font='Times New Roman',
                  title_font_size=15, axis_font_size=12, labelsize=12,
                  filename=None, file_format='png', dpi=300):
        '''Test Information Curve.'''
        difficulties, thresholds, severities = self._get_params(model, anchor)
        if isinstance(items, str) and items in ('all', 'none'): items = None
        if isinstance(raters, str) and raters in ('all', 'none', 'zero'): raters = None
        if isinstance(items, str):
            items = None if items in ('all', 'none') else [items]
        if isinstance(raters, str):
            raters = None if raters in ('all', 'none', 'zero') else [raters]
        item_keys  = list(self.items) if items is None else items
        rater_list = list(self.raters) if raters is None else raters

        abilities = np.arange(-20, 20, 0.1)
        y = np.array([
            sum(self.variance(a, it, difficulties, r, severities, thresholds, model)
                for it in item_keys for r in rater_list)
            for a in abilities
        ]).reshape(-1, 1)
        if ymax is None:
            ymax = float(y.max()) * 1.1

        return self.plot_data(
            x_data=abilities, y_data=y, model=model, anchor=anchor,
            items=items, raters=raters, x_min=xmin, x_max=xmax, y_max=ymax,
            graph_title=title or '', point_info_lines_test=point_info_lines,
            score_labels=point_info_labels, y_label='Fisher information',
            plot_style=plot_style, palette=palette, black=black, font=font,
            title_font_size=title_font_size, axis_font_size=axis_font_size,
            labelsize=labelsize, filename=filename, plot_density=dpi,
            file_format=file_format
        )

    def test_info_global(self, **kw):     return self.test_info(model='global', **kw)
    def test_info_items(self, **kw):      return self.test_info(model='items', **kw)
    def test_info_thresholds(self, **kw): return self.test_info(model='thresholds', **kw)
    def test_info_matrix(self, **kw):     return self.test_info(model='matrix', **kw)

    def test_csem(self, model='global', anchor=False, items=None, raters=None,
                  point_csem_lines=None, point_csem_labels=False,
                  xmin=-5, xmax=5, ymax=5, title=None, plot_style='white',
                  palette='dark blue', black=False, font='Times New Roman',
                  title_font_size=15, axis_font_size=12, labelsize=12,
                  filename=None, file_format='png', dpi=300):
        '''Test Conditional Standard Error of Measurement Curve.'''
        difficulties, thresholds, severities = self._get_params(model, anchor)
        if isinstance(items, str) and items in ('all', 'none'): items = None
        if isinstance(raters, str) and raters in ('all', 'none', 'zero'): raters = None
        if isinstance(items, str):
            items = None if items in ('all', 'none') else [items]
        if isinstance(raters, str):
            raters = None if raters in ('all', 'none', 'zero') else [raters]
        item_keys  = list(self.items) if items is None else items
        rater_list = list(self.raters) if raters is None else raters

        abilities = np.arange(-20, 20, 0.1)
        info = np.array([
            sum(self.variance(a, it, difficulties, r, severities, thresholds, model)
                for it in item_keys for r in rater_list)
            for a in abilities
        ])
        y = (1.0 / (info ** 0.5)).reshape(-1, 1)

        return self.plot_data(
            x_data=abilities, y_data=y, model=model, anchor=anchor,
            items=items, raters=raters, x_min=xmin, x_max=xmax, y_max=ymax,
            graph_title=title or '', point_csem_lines=point_csem_lines,
            score_labels=point_csem_labels, y_label='Conditional SEM',
            plot_style=plot_style, palette=palette, black=black, font=font,
            title_font_size=title_font_size, axis_font_size=axis_font_size,
            labelsize=labelsize, filename=filename, plot_density=dpi,
            file_format=file_format
        )

    def test_csem_global(self, **kw):     return self.test_csem(model='global', **kw)
    def test_csem_items(self, **kw):      return self.test_csem(model='items', **kw)
    def test_csem_thresholds(self, **kw): return self.test_csem(model='thresholds', **kw)
    def test_csem_matrix(self, **kw):     return self.test_csem(model='matrix', **kw)

    def std_residuals_plot(self, model='global', items=None, raters=None,
                           bin_width=0.5, x_min=-6, x_max=6, normal=False,
                           title=None, plot_style='white', font='Times New Roman',
                           title_font_size=15, axis_font_size=12, labelsize=12,
                           filename=None, file_format='png', plot_density=300):
        '''Standardised residuals histogram with optional item/rater subsetting.'''
        if not hasattr(self, f'std_residual_df_{model}'):
            self.fit_statistics(model=model)

        std_res = getattr(self, f'std_residual_df_{model}')

        # Normalise string arguments
        if isinstance(raters, str):
            if raters in ('all', 'none'):
                raters = None
            else:
                raters = [raters]
        if isinstance(items, str):
            if items in ('all', 'none'):
                items = None
            else:
                items = [items]

        # Subset
        if items is None and raters is None:
            residuals = pd.Series(std_res.values.flatten()).dropna()
        elif items is None:
            residuals = pd.Series(std_res.loc[raters].values.flatten()).dropna()
        elif raters is None:
            residuals = pd.Series(std_res[items].values.flatten()).dropna()
        else:
            residuals = pd.Series(std_res[items].loc[raters].values.flatten()).dropna()

        return self.std_residuals_hist(
            residuals, bin_width=bin_width, x_min=x_min, x_max=x_max,
            normal=normal, title=title, plot_style=plot_style, font=font,
            title_font_size=title_font_size, axis_font_size=axis_font_size,
            labelsize=labelsize, filename=filename, file_format=file_format,
            plot_density=plot_density
        )

    def std_residuals_plot_global(self, **kw):     return self.std_residuals_plot(model='global', **kw)
    def std_residuals_plot_items(self, **kw):      return self.std_residuals_plot(model='items', **kw)
    def std_residuals_plot_thresholds(self, **kw): return self.std_residuals_plot(model='thresholds', **kw)
    def std_residuals_plot_matrix(self, **kw):     return self.std_residuals_plot(model='matrix', **kw)