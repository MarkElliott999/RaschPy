from math import log
import warnings
from scipy.stats import chi2, gaussian_kde, norm, t as t_dist

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from matplotlib import colors as colors
from matplotlib import cm as cmx
from matplotlib.patches import Rectangle
import seaborn as sns
import matplotlib.patheffects as path_effects

from raschpy.base import Rasch


class RSM(Rasch):
    """
    Rating Scale Model (Andrich 1978) formulation of the polytomous Rasch model.

    The RSM constrains all items to share the same set of Rasch-Andrich
    threshold parameters (tau_1..tau_m), differing only in their central
    item locations (delta_i). Thresholds are estimated using CPAT
    (Elliott & Buttery, 2022); item locations are estimated using PAIR.

    Threshold convention: self.thresholds is a numpy array of length max_score,
    where thresholds[0..max_score-1] are the Rasch-Andrich threshold parameters
    tau_1..tau_m.
    """

    def __init__(
        self,
        responses,
        max_score=None,
        extreme_persons=True,
        no_of_classes=5,
        validate=True,
        exogenous=None,
    ):
        """
        Initialise a Rating Scale Model object.

        Parameters
        ----------
        responses : pandas.DataFrame or RSM_Sim
            Response data with persons as rows and items as columns.
            Cell values should be integers in [0, max_score] or NaN for
            missing. Alternatively, pass an RSM_Sim object to instantiate
            directly from a simulation; generating parameters are stored
            in self.generating.
        max_score : int or None, default None
            Maximum possible score per item (shared across all items).
            If None, inferred from the observed data maximum. Must not
            be less than the observed maximum.
        extreme_persons : bool, default True
            If True, removes only persons with entirely missing data.
            If False, additionally removes persons with all-zero or
            perfect total scores, which cannot be estimated by ML.
        no_of_classes : int, default 5
            Number of class intervals used in observed-data overlays on
            ICC, CRC, and TCC plots.
        validate : bool, default True
            If True, checks whether the item response network is fully
            connected (i.e. all items are linked via common persons).
            Issues a UserWarning if the data is split into disconnected
            sub-networks, which makes item locations incomparable
            across sub-groups.
        exogenous : pandas.DataFrame or None, default None
            Optional person-level covariates (e.g. Gender, L1) for
            differential item functioning analysis, indexed by person
            identifier. Values are kept as raw category labels. Persons
            in responses without a matching exogenous record (and vice
            versa) are allowed — such gaps are common when exogenous
            data is optional (e.g. for GDPR reasons) — and are reported
            via UserWarning plus the exogenous_only_persons /
            no_exogenous_persons attributes, rather than raising.

        Attributes set
        --------------
        responses : pandas.DataFrame
            Filtered response data (invalid/extreme persons removed).
        invalid_responses : pandas.DataFrame
            Rows removed due to all-NaN response patterns.
        extreme_persons : pandas.DataFrame
            Rows removed due to extreme scores (if extreme_persons=False).
        max_score : int
            Maximum possible score per item.
        no_of_items : int
            Number of items in the filtered responses.
        no_of_persons : int
            Number of persons in the filtered responses.
        item_names : pandas.Index
            Item identifiers (column names of responses).
        person_names : pandas.Index
            Person identifiers (index of responses).
        no_of_classes : int
            Number of class intervals (passed through for plot methods).
        connectivity_status : dict
            Result of check_data_connectivity(), present only if
            validate=True. Contains at minimum a 'connected' key (bool)
            and 'components_count' (int).
        exogenous : pandas.DataFrame or None
            Person-level covariates reindexed onto person_names, or None
            if not supplied.
        no_exogenous_persons : pandas.Index
            Persons present in responses with no matching exogenous record.
        exogenous_only_persons : pandas.Index
            Persons present in the exogenous data but not in responses.
        """

        # Sim-aware instantiation: store sim attributes in self.generating namespace
        from raschpy.simulation.rsm_sim import RSM_Sim
        from raschpy.base import _SimParams

        if isinstance(responses, RSM_Sim):
            sim = responses
            self.generating = _SimParams()
            for attr, value in vars(sim).items():
                setattr(self.generating, attr, value)
            if max_score is not None and max_score != sim.max_score:
                warnings.warn(
                    f"max_score={max_score} does not match sim.max_score={sim.max_score}. "
                    f"Using max_score={max_score}."
                )
                self.max_score = int(max_score)
            else:
                self.max_score = int(sim.max_score)
            responses = sim.responses
        else:
            self.max_score = (
                int(np.nanmax(responses)) if max_score is None else int(max_score)
            )

        # Validate max_score against observed data
        observed_max = int(np.nanmax(responses))
        if self.max_score < observed_max:
            raise ValueError(
                f"max_score={self.max_score} is less than the maximum observed score "
                f"({observed_max}) in the data."
            )
        if self.max_score > observed_max:
            warnings.warn(
                f"max_score={self.max_score} exceeds the maximum observed score "
                f"({observed_max}) in the data. Some score categories may be unobserved."
            )

        # Always remove all-NaN rows (truly invalid — no usable data)
        all_nan_mask = responses.isna().all(axis=1)
        self.invalid_responses = responses[all_nan_mask]
        valid = responses[~all_nan_mask]

        if extreme_persons:
            self.extreme_persons = valid.iloc[
                0:0
            ]  # empty; no persons removed as extreme
            self.responses = valid
        else:
            scores = valid.sum(axis=1)
            # notna() mask is cleaner than (df == df) for detecting valid cells
            max_scores = valid.notna().sum(axis=1) * self.max_score
            extreme_mask = (scores == 0) | (scores == max_scores)
            self.extreme_persons = valid[extreme_mask]
            self.responses = valid[~extreme_mask]

        self.no_of_items = self.responses.shape[1]
        self.item_names = self.responses.columns
        self.no_of_persons = self.responses.shape[0]
        self.person_names = self.responses.index
        self.no_of_classes = no_of_classes

        # Optional person-level covariates for DIF (e.g. Gender, L1)
        if exogenous is not None:
            self.no_exogenous_persons = self.person_names[
                ~self.person_names.isin(exogenous.index)
            ]
            self.exogenous_only_persons = exogenous.index[
                ~exogenous.index.isin(self.person_names)
            ]
            self.exogenous = exogenous.reindex(self.person_names)

            if len(self.no_exogenous_persons) > 0:
                warnings.warn(
                    f"{len(self.no_exogenous_persons)} person(s) in the response data "
                    f"have no matching exogenous record (exogenous data is often "
                    f"optional, e.g. for GDPR reasons). See no_exogenous_persons for "
                    f"the full list. These persons will be excluded from any DIF "
                    f"grouping that relies on the missing covariate(s).",
                    UserWarning,
                    stacklevel=2,
                )

            if len(self.exogenous_only_persons) > 0:
                warnings.warn(
                    f"{len(self.exogenous_only_persons)} person(s) in the exogenous "
                    f"data are not present in the response data and will be ignored. "
                    f"See exogenous_only_persons for the full list.",
                    UserWarning,
                    stacklevel=2,
                )
        else:
            self.exogenous = None
            self.no_exogenous_persons = pd.Index([])
            self.exogenous_only_persons = pd.Index([])

        # RUN AUTOMATIC CONNECTION CHECK VALIDATION
        if validate:
            self.connectivity_status = self.check_data_connectivity()

            # THROW SYSTEM WARNING WITH MATHEMATICAL DETAILS IF DISCONNECTED
            if not self.connectivity_status["connected"]:
                warnings.warn(
                    f"\n"
                    f"⚠️  CRITICAL DATA INTEGRITY WARNING: The response data is disconnected into "
                    f"{self.connectivity_status['components_count']} separate sub-networks.\n"
                    f"Item location estimates will be problematic because there are no empirical "
                    f"comparisons spanning across these isolated groups, the item parameter "
                    f"estimates for each independent subset will separately sum to zero. This means items "
                    f"belonging to different subsets cannot be compared or calibrated onto a single scale.",
                    category=UserWarning,
                    stacklevel=2,
                )

            # THROW SYSTEM WARNING FOR "FAKE CONNECTIVITY" — ITEMS THAT PASS THE
            # STANDARD CHECK BUT WILL STILL BREAK CALIBRATE()'S DIRECTED MATRIX
            directionally_isolated = self.connectivity_status.get(
                "directionally_isolated_items", []
            )
            if directionally_isolated:
                warnings.warn(
                    f"\n"
                    f"⚠️  DATA INTEGRITY WARNING: {len(directionally_isolated)} item(s) have a "
                    f"structurally unresolvable zero in calibrate()'s directed pairwise matrix: "
                    f"{directionally_isolated}.\n"
                    f"These items pass the standard connectivity check (they have at least one "
                    f"empirical comparison in some direction) but can silently produce NaN "
                    f"or overflow during calibration rather than a clear error. Consider dropping "
                    f"these items or gathering more responses before calibrating.",
                    category=UserWarning,
                    stacklevel=2,
                )

    # ------------------------------------------------------------------
    # Core probability / expected-score functions (scalar, used in plots)
    # ------------------------------------------------------------------

    def cat_prob(self, person_location, item_location, category, thresholds):
        """
        Compute the probability of a response category (centred RSM parameterisation).

        Log-numerator for category k: k*(b-d) - cumsum(tau)[k], where b is person location,
        d is item location, tau[0]=0 sentinel. Numerically stabilised via log-sum-exp.

        Parameters
        ----------
        person_location : float
            Person location on the logit scale.
        item_location : float
            Central item location on the logit scale.
        category : int
            Response category (0 to max_score).
        thresholds : array-like
            Rasch-Andrich threshold vector, length max_score, centred at 0.

        Returns
        -------
        float
            Probability of the specified category, in [0, 1].
        """
        cats = np.arange(len(thresholds) + 1, dtype=float)
        cumsum = np.concatenate([[0.0], np.cumsum(thresholds)])
        log_nums = cats * (person_location - item_location) - cumsum
        log_nums -= log_nums.max()
        nums = np.exp(log_nums)
        return nums[category] / nums.sum()

    def exp_score(self, person_location, item_location, thresholds):
        """
        Compute the expected score on an item.

        Calculates E[X | person location, item location, thresholds] = sum(k * P(X=k))
        using the RSM centred parameterisation. Numerically stabilised.

        Parameters
        ----------
        person_location : float
            Person location on the logit scale.
        item_location : float
            Item location on the logit scale.
        thresholds : array-like
            Rasch-Andrich threshold vector, length max_score.

        Returns
        -------
        float
            Expected score in [0, max_score].
        """
        cats = np.arange(len(thresholds) + 1, dtype=float)
        cumsum = np.concatenate([[0.0], np.cumsum(thresholds)])
        log_nums = cats * (person_location - item_location) - cumsum
        log_nums -= log_nums.max()
        nums = np.exp(log_nums)
        probs = nums / nums.sum()
        return (cats * probs).sum()

    def variance(self, person_location, item_location, thresholds):
        """
        Compute item variance (Fisher information).

        Calculates Var[X | person location, item location, thresholds] = sum((k - E[X])^2 * P(X=k)).
        Equal to the Fisher information at the given person location.

        Parameters
        ----------
        person_location : float
            Person location on the logit scale.
        item_location : float
            Item location on the logit scale.
        thresholds : array-like
            Rasch-Andrich threshold vector, length max_score.

        Returns
        -------
        float
            Item variance / Fisher information. Always non-negative.
        """
        cats = np.arange(len(thresholds) + 1, dtype=float)
        cumsum = np.concatenate([[0.0], np.cumsum(thresholds)])
        log_nums = cats * (person_location - item_location) - cumsum
        log_nums -= log_nums.max()
        nums = np.exp(log_nums)
        probs = nums / nums.sum()
        expected = (cats * probs).sum()
        return ((cats - expected) ** 2 * probs).sum()

    def kurtosis(self, person_location, item_location, thresholds):
        """
        Compute the fourth central moment of the response distribution.

        Calculates sum((k - E[X])^4 * P(X=k)) using the RSM centred
        parameterisation. Used in the Wilson-Hilferty approximation for
        standardised fit statistics (Infit Z, Outfit Z).

        Parameters
        ----------
        person_location : float
            Person location on the logit scale.
        item_location : float
            Item location on the logit scale.
        thresholds : array-like
            Rasch-Andrich threshold vector, length max_score.

        Returns
        -------
        float
            Fourth central moment of the response distribution.
        """
        cats = np.arange(len(thresholds) + 1, dtype=float)
        cumsum = np.concatenate([[0.0], np.cumsum(thresholds)])
        log_nums = cats * (person_location - item_location) - cumsum
        log_nums -= log_nums.max()
        nums = np.exp(log_nums)
        probs = nums / nums.sum()
        expected = (cats * probs).sum()
        return ((cats - expected) ** 4 * probs).sum()

    # ------------------------------------------------------------------
    # Vectorised category probability engine
    # ------------------------------------------------------------------

    def _cat_probs_matrix(self, person_locations, item_locations, thresholds):
        """
        Vectorised RSM category probability computation.

        The RSM log-numerator for category k, person n, item i is:
            k * (person_location_n - item_location_i) - cumsum(thresholds)[k]
        where cumsum(thresholds)[k] = sum(thresholds[0..k]).

        Because thresholds are SHARED across items (unlike PCM), cumsum is
        identical for all items and the full (K+1, N, I) tensor is computed
        in a single broadcast without any Python loop over items or categories.

        Returns
        -------
        probs    : ndarray (K+1, N, I)  -- category probabilities
        cats_arr : ndarray (K+1,)       -- category indices [0..max_score]
        """
        cats_arr = np.arange(len(thresholds) + 1, dtype=float)  # (K+1,)
        thr_arr = np.asarray(thresholds, dtype=float)
        cumsum = np.concatenate([[0.0], np.cumsum(thr_arr)])  # (K+1,)
        ab = np.asarray(person_locations, dtype=float)  # (N,)
        diff = np.asarray(item_locations, dtype=float)  # (I,)

        # log_num[k, n, i] = k*(ab[n] - diff[i]) - cumsum[k]
        log_num = (
            cats_arr[:, None, None] * (ab[None, :, None] - diff[None, None, :])
            - cumsum[:, None, None]
        )  # (K+1, N, I)

        # Numerically stable softmax along category axis
        log_num -= log_num.max(axis=0, keepdims=True)
        probs = np.exp(log_num)
        probs /= probs.sum(axis=0, keepdims=True)

        return probs, cats_arr

    # ------------------------------------------------------------------
    # CPAT threshold estimation
    # ------------------------------------------------------------------

    def _threshold_distance(self, threshold, item_locations, constant=0.1):
        """
        Estimate the distance between adjacent Rasch-Andrich thresholds (CPAT).

        Implements Elliott & Buttery (2022). For threshold k, counts:
          num[i,j]: persons scoring exactly k on both items i and j
          den[i,j]: persons scoring k-1 on item i and k+1 on item j
        Conditioning on these patterns removes person location, leaving a
        contrast identifying the threshold location. Harmonic mean weighting
        downweights near-zero counts. Vectorised via matrix multiplication.

        Parameters
        ----------
        threshold : int
            1-based threshold index (1 to max_score-1).
        item_locations : pandas.Series
            Item location estimates indexed by item name.
        constant : float, default 0.1
            Additive smoothing constant for zero cells.

        Returns
        -------
        float
            Estimated distance between tau_threshold and tau_{threshold+1}, in logits.

        Raises
        ------
        ValueError
            If no item pair has any adjacent-category co-occurrence for this
            threshold (no information anywhere in the data to estimate it).

        Notes
        -----
        A cell (item pair) with no co-occurrence at all (num == den == 0)
        carries no information and is dropped, as always. At ``constant=0``
        a cell where only ONE side was observed (num == 0 xor den == 0) is
        ALSO dropped — its log-ratio is undefined without smoothing, and
        letting ``log(0) = -inf`` survive multiplication by its own
        (correctly) zero weight would silently poison the whole weighted
        average via ``0 * -inf = nan``. For ``constant > 0`` this changes
        nothing: `num_s`/`den_s` are always > 0 wherever a pair is otherwise
        valid.
        """
        df_array = np.array(self.responses, dtype=np.float64)

        # Build (N, I) indicator arrays for each relevant score value
        at_k = (df_array == threshold).astype(np.float64)  # X == k
        at_km1 = (df_array == threshold - 1).astype(np.float64)  # X == k-1
        at_kp1 = (df_array == threshold + 1).astype(np.float64)  # X == k+1

        # (I, I) count matrices via matrix multiplication
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            num_matrix = at_k.T @ at_k  # count(X_i==k AND X_j==k)
            den_matrix = at_km1.T @ at_kp1  # count(X_i==k-1 AND X_j==k+1)

        valid = (num_matrix + den_matrix) > 0
        num_s = np.where(valid, num_matrix + constant, 0.0)
        den_s = np.where(valid, den_matrix + constant, 0.0)
        finite = valid & (num_s > 0) & (den_s > 0)

        # Location contrast matrix: delta_i - delta_j  shape (I, I)
        estimates = item_locations.values
        item_location_matrix = estimates[:, None] - estimates[None, :]

        with np.errstate(divide="ignore", invalid="ignore"):
            safe_num = np.where(finite, num_s, 1.0)
            safe_den = np.where(finite, den_s, 1.0)
            log_ratio = np.log(safe_num) - np.log(safe_den)
            # Harmonic mean weight: 2*a*b/(a+b), zero where not finite
            weight_matrix = np.where(finite, 2.0 * num_s * den_s / (num_s + den_s), 0.0)

        total_weight = weight_matrix.sum()
        if total_weight == 0:
            raise ValueError(
                f"CPAT threshold distance for threshold {threshold} could not be "
                f"estimated: no pair of items has any adjacent-category "
                f"co-occurrence at constant={constant}. Use a larger sample, "
                f"a non-zero `constant`, or drop this threshold from the data."
            )

        return (weight_matrix * (log_ratio + item_location_matrix)).sum() / total_weight

    def threshold_set(self, item_locations, constant=0.1):
        """
        Compute the Rasch-Andrich threshold vector from CPAT distances.

        Chains the m-1 CPAT distances via cumulative sum and mean-centres the
        result. Called by calibrate().

        Parameters
        ----------
        item_locations : pandas.Series
            Item locations from Stage 1 (PAIR), indexed by item name.
        constant : float, default 0.1
            Additive smoothing constant passed to _threshold_distance() for
            every adjacent-threshold link.

        Returns
        -------
        numpy.ndarray
            Threshold vector of length max_score, centred at 0.
        """
        thresh_distances = [
            self._threshold_distance(threshold + 1, item_locations, constant)
            for threshold in range(self.max_score - 1)
        ]

        # Chain distances: thresh[k] = sum of distances[0..k-1]
        thresholds = np.array(
            [sum(thresh_distances[:t]) for t in range(self.max_score)]
        )
        thresholds -= np.mean(thresholds)
        return thresholds

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def _build_pairwise_matrix(self):
        """
        Raw (unsmoothed) directed pairwise comparison matrix used by
        calibrate() and check_data_connectivity(). Entry (i, j) counts
        persons who scored exactly one point higher on item i than item j.

        Returns
        -------
        matrix : numpy.ndarray, shape (no_of_items, no_of_items)
        row_items : numpy.ndarray
            Item name for each row/column (identity mapping for RSM).
        """
        df_array = self.responses.to_numpy(dtype=np.float64)
        matrix = np.zeros((self.no_of_items, self.no_of_items), dtype=np.float64)
        for category in range(self.max_score):
            higher = (df_array == category + 1).astype(np.float64)
            lower = (df_array == category).astype(np.float64)
            matrix += higher.T @ lower
        return matrix, np.array(self.item_names)

    def calibrate(
        self, constant=None, threshold_constant=None, method="log-lik",
        matrix_power=None, log_lik_tol=0.000001):
        """
        Two-stage RSM calibration: PAIR item locations + CPAT thresholds.

        Stage 1: Builds a pairwise contingency matrix (entry (i,j) = count of
        persons scoring one point higher on item i than j), resolves structural
        zeroes via matrix powers, and extracts item locations with
        priority_vector().

        Stage 2: Given item locations, estimates adjacent-threshold distances
        via CPAT (_threshold_distance()), chains them, and centres the result.

        Issues a UserWarning if only one item is present or if the resolved
        Stage 1 constant is 0 and any item has all-maximum scores.

        Parameters
        ----------
        constant : float or None, default None
            Additive smoothing constant for the Stage 1 (item pairwise) matrix.
            ``None`` resolves to 0.1. If ``threshold_constant`` is left ``None``
            it also sets the Stage 2 (CPAT threshold) constant: to 0.03 when
            ``constant`` is ``None``, or to the same explicit value otherwise.
            Use 0 to disable Stage 1 smoothing (item estimation then fails if
            any item has all-maximum scores).
        threshold_constant : float or None, default None
            Additive smoothing constant for the Stage 2 CPAT threshold
            distances. ``None`` follows ``constant`` (0.03 when ``constant`` is
            ``None``, else the explicit ``constant`` value); pass a float to
            set the threshold stage independently of the item stage. The
            default threshold value is smaller than the item value because
            CPAT already regularises through its harmonic-mean information
            weighting -- thin comparison cells contribute little regardless of
            the additive term -- so the constant only has to keep the
            log-ratios finite. A large scalar sweep over well-conditioned,
            edge (tight scales / small N / missing data) and
            disordered-threshold designs put the threshold-recovery RMSE and
            slope-bias optimum at or below 0.03 in the edge and disordered
            regimes, with the well-conditioned curve flat to within ~1% across
            [0.03, 0.15].
        method : str, default 'log-lik'
            Priority vector extraction method.
        matrix_power : int or None, default None
            Initial matrix power before checking for structural zeroes.
            None resolves to 0 for method='log-lik' (no powering), else 5.
        log_lik_tol : float, default 0.000001
            Log-likelihood convergence tolerance for priority vector extraction.

        Attributes set
        --------------
        items : pandas.Series
            Item location estimates, zero-centred.
        thresholds : pandas.Series
            Shared Rasch-Andrich threshold vector, length max_score,
            centred at 0.
        constant : float
            Stage 1 (item) smoothing constant actually used.
        threshold_constant : float
            Stage 2 (CPAT threshold) smoothing constant actually used.
        null_persons : pandas.Index
            Persons dropped (entirely missing data).
        """

        if len(self.responses.columns) == 1:
            warnings.warn(
                "Only one item detected. This model with a single item reduces to RSM "
                "with raters as items. Consider reconfiguring and using RSM instead.",
                UserWarning,
                stacklevel=2,
            )

        # Resolve the two-stage smoothing constants:
        #   constant=None                     -> item 0.1
        #   constant=c                        -> item c
        #   threshold_constant=None           -> threshold follows `constant`
        #                                        (0.03 if constant is None, else c)
        #   threshold_constant=t              -> threshold t (independent)
        item_constant = 0.1 if constant is None else float(constant)
        if threshold_constant is not None:
            thresh_constant = float(threshold_constant)
        elif constant is None:
            thresh_constant = 0.03
        else:
            thresh_constant = float(constant)
        self.constant = item_constant
        self.threshold_constant = thresh_constant

        if item_constant == 0:
            # Unsmoothed PAIR: re-run the structural connectivity check on the
            # data as it stands now (not just at __init__(validate=True)) --
            # constant=0 removes the only thing masking a directionally
            # isolated item or disconnected component, both of which
            # check_data_connectivity() already detects and warns about
            # (does not raise; 'log-lik' tolerates an incomplete comparison
            # graph and still returns an answer for the rest of the items).
            self.check_data_connectivity()

        self.null_persons = self.responses.index[self.responses.isnull().all(1)]
        self.responses = self.responses.drop(self.null_persons)
        self.no_of_persons = self.responses.shape[0]

        matrix, _ = self._build_pairwise_matrix()
        matrix_power = self._resolve_matrix_power(method, matrix_power)

        constant_matrix = ((matrix + matrix.T) > 0).astype(np.float64) * item_constant
        matrix += constant_matrix
        np.fill_diagonal(matrix, matrix.diagonal() + item_constant)

        # Sparse/disconnected resamples can blow this up to inf/nan before the zero-check
        # loop below terminates; not a real numerical error, so suppress the noise.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            if matrix_power == 0:
                # No powering: 'log-lik' (Bradley-Terry) consumes structural
                # zeroes directly. Fill remaining off-diagonal zeroes
                # (unobserved pairs) with `constant`.
                mat = np.array(matrix, dtype=np.float64)
                if item_constant:
                    off_diagonal_mask = ~np.eye(self.no_of_items, dtype=bool)
                    mat[off_diagonal_mask & (mat == 0)] = item_constant
            else:
                mat = np.linalg.matrix_power(matrix, matrix_power)
                mat_pow = matrix_power
                while 0 in mat:
                    mat = mat @ matrix
                    mat_pow += 1
                    if mat_pow == matrix_power + 5:
                        mat += item_constant
                        break

        self.items = self.priority_vector(mat, method=method, log_lik_tol=log_lik_tol)

        # Stage 2: CPAT threshold estimation
        self.thresholds = pd.Series(
            self.threshold_set(self.items, constant=thresh_constant),
            index=range(1, self.max_score + 1),
        )

    # ------------------------------------------------------------------
    # Anchor calibration
    # ------------------------------------------------------------------

    def calibrate_anchor(
        self,
        anchors,
        calibrate=False,
        selection_method="robust_z",
        corr_tol=0.95,
        sd_ratio_tol=1.1,
        min_anchors=6,
        wald_alpha=0.05,
        no_of_samples=500,
        adj=None,
        overwrite_anchors="none",
        plot=True,
        plot_kwargs=None,
        anchor_thresholds=None,
        alpha=0.05,
        correction="bh",
        warm_corr=True,
        tolerance=0.00001,
        max_iters=100,
        ext_score_adjustment=0.5,
        constant=None,
        threshold_constant=None,
        method="log-lik",
        matrix_power=None,
        log_lik_tol=0.000001,
        seed=None,
    ):
        """
        Anchor item location estimates onto externally-supplied values.

        Supports item banking: calibrates this dataset's own item
        locations as usual, then shifts the whole item-location scale
        by a translation constant so that a subset of common ("anchor")
        items line up with externally-supplied reference locations (e.g.
        from a bank of previously-calibrated items). This is a translation
        only (RSM item discrimination is fixed at 1 across items), exactly
        as in SLM.

        The shared threshold vector (self.thresholds) is left untouched by
        anchoring itself. Thresholds describe the relative category
        structure — common to all items — not an item's location on the
        logit scale, so an item-bank shift doesn't apply to them: adding a
        constant to every item location and re-estimating person locations
        against the same (unshifted) thresholds reproduces exactly the same
        category probabilities and fit statistics as before anchoring, just
        relocated on the shared scale. There is deliberately no
        anchor_thresholds attribute holding a *shifted* version of
        self.thresholds.

        Optionally (anchor_thresholds=), if you also have an externally-supplied reference threshold
        structure (e.g. the bank's own category structure), this runs two
        checks of whether it's actually compatible with this dataset.

        The first is an omnibus likelihood-ratio test: holding item
        locations fixed at anchor_items, it compares the model fit (log-
        likelihood, person locations re-estimated in each case) using this
        dataset's own freely-estimated thresholds (self.thresholds) against
        using the given anchor_thresholds instead. (AIC/BIC don't apply
        here: both candidates are point-values plugged into the exact same
        threshold-vector-length model, so there's no genuine difference in
        the number of model parameters to penalise for — only the LR
        test's fixed-vs-free chi-squared framing is actually appropriate.)
        Because person locations are re-estimated freely under each
        candidate, this comparison is already invariant to any overall
        shift in the threshold vector's level (a uniform shift is absorbed
        exactly by the re-estimated person locations, the standard
        Rasch θ/δ indeterminacy) — so despite operating on the raw
        threshold vectors, it only ever detects genuine differences in
        category *shape* (widths), never a harmless difference in
        centring convention between this dataset and the anchor source.
        A significantly worse fit under anchor_thresholds warns that the
        rating-scale structures may not actually be shared, which would
        undermine a translation-only equating assumption.

        The second is a per-category test, run alongside the omnibus one:
        this dataset's own bootstrap category widths (self.thresholds.diff(),
        with SEs from cat_width_se — auto-triggers std_errors() if not
        already computed) are compared directly against anchor_thresholds'
        own widths via a Wald z-test, one per category. Unlike the omnibus
        test, this localises *which* category's shape differs, the same
        way dif_test's category-width (DCF) test does across groups —
        comparing raw threshold locations here would be the wrong quantity,
        since a genuine difference in one category's width cascades
        additively into every threshold location after it, smearing one
        true effect across several per-category tests.

        By default (selection_method='robust_z'), the translation constant
        is computed via _robust_anchor_selection() (Iglewicz & Hoaglin
        modified z-score against the anchor set's own median/MAD), which
        iteratively excludes anchor items whose calibrated value has
        drifted too far from its supplied reference value, so a handful of
        stale or misbehaving anchor items cannot distort the shift applied
        to the rest of the scale. selection_method='wald' is an
        alternative using a formal significance test instead of a
        descriptive-statistics trim (see _wald_anchor_selection). Set
        selection_method='none' for a plain mean shift over all supplied
        anchors, no trimming.

        Parameters
        ----------
        anchors : dict or pandas.Series
            Externally-supplied reference item locations, keyed/indexed by
            item name. Only items also present in this dataset are used.
        calibrate : bool, default False
            If True, (re-)runs calibrate() before anchoring. If False,
            calibrate() is still auto-triggered if self.items does not
            yet exist.
        selection_method : {'robust_z', 'wald', 'none'}, default 'robust_z'
            'robust_z': _robust_anchor_selection() — Iglewicz & Hoaglin
            modified z-score / MAD-based iterative trim (corr_tol,
            sd_ratio_tol, min_anchors).
            'wald': _wald_anchor_selection() — a genuine significance test
            per anchor item (z = (anchor - (observed + tc)) / SE(observed),
            tc precision-weighted, sequential exclusion) rather than a
            descriptive-statistics trim. Needs bootstrap item SEs — auto-
            triggers std_errors() if not already computed. Uses wald_alpha
            and min_anchors, not corr_tol/sd_ratio_tol.
            'none': plain mean shift over every supplied anchor, no
            trimming at all.
        corr_tol : float, default 0.95
            Passed to _robust_anchor_selection() (selection_method=
            'robust_z' only).
        sd_ratio_tol : float, default 1.1
            Passed to _robust_anchor_selection() (selection_method=
            'robust_z' only).
        min_anchors : int, default 6
            Floor on surviving anchor items — passed to whichever
            selection method is active ('robust_z' or 'wald').
        wald_alpha : float, default 0.05
            Significance level for each anchor item's Wald test
            (selection_method='wald' only).
        no_of_samples : int, default 500
            Bootstrap samples for std_errors(), only used if item SEs
            aren't already computed (selection_method='wald' only).
        seed : int or None, default None
            Seed passed through to the internal std_errors() call (only
            used if item SEs aren't already computed). None draws fresh
            entropy each call.
        adj : float or None, default None
            If provided, this translation constant is applied directly and
            the selection step is skipped entirely. Intended for reuse
            across bootstrap resamples, where recomputing anchor selection
            on every resample would otherwise inflate standard errors with
            anchor-item sampling variance.
        overwrite_anchors : 'none', 'rejected', or 'all', default 'none'
            Controls what anchor_items holds for the anchor items
            themselves (any item present in both anchors and this
            dataset), as opposed to non-anchor items, which always get
            the shifted (calibrated + translation constant) value.
            'none' (default): every anchor item keeps exactly its
            externally-supplied value from anchors, unchanged — the usual
            convention for a genuine anchor (its value is fixed by
            definition, not re-estimated).
            'rejected': anchors kept by selection still keep their exact
            supplied value, but anchors rejected as outliers instead
            receive their own shifted, freshly-calibrated value — useful
            when a rejected anchor's supplied value is itself suspected to
            be stale or wrong, so you'd rather trust this dataset's own
            estimate for that specific item. Requires selection_method in
            ('robust_z', 'wald') and no adj override; otherwise there is
            no selected/rejected distinction to act on, and this falls
            back to 'none' with a warning.
            'all': every anchor item receives its own shifted value like
            any other item, overwriting the supplied anchor value with
            what this dataset actually observed.
        plot : bool, default True
            If True and selection_method in ('robust_z', 'wald') (so a
            selection table exists), calls plot_anchor_selection()
            automatically at the end. Has no effect when
            selection_method='none' or adj is supplied directly, since no
            selection table is produced in those cases.
        plot_kwargs : dict or None, default None
            Extra keyword arguments forwarded to plot_anchor_selection()
            when plot=True (e.g. filename, xmin/xmax, title).
        anchor_thresholds : array-like or None, default None
            Externally-supplied reference threshold vector (length
            max_score), e.g. from the same bank the item anchors came from.
            If supplied, runs the LR comparison described above and warns
            if the two threshold structures are significantly different.
            If None, the threshold-structure check is skipped entirely
            (self.anchor_threshold_test is set to None).
        alpha : float, default 0.05
            Significance level for both the omnibus likelihood-ratio test
            (self.thresholds, freely estimated, df=max_score-1, vs
            anchor_thresholds, fixed, df=0, chi-squared with
            df=max_score-1) and the per-category Wald z-tests (compared
            against the corrected p-value if correction is set, else the
            raw p-value).
        correction : {'bh', 'bonferroni', None}, default 'bh'
            Multiple-comparison correction across the per-category width
            tests only (mirrors dif_test's convention elsewhere in the
            package). None uses raw p-values, uncorrected. Not applied to
            the omnibus test, which is already a single test.
        warm_corr, tolerance, max_iters, ext_score_adjustment :
            Person-estimation kwargs, used only by the omnibus
            anchor_thresholds comparison (person locations must be
            re-estimated under each candidate threshold vector before the
            log-likelihoods are comparable).
        constant, method, matrix_power, log_lik_tol : floats
            Calibration kwargs, used only if calibrate is triggered, or if
            cat_width_se needs to be computed for the per-category test.

        Attributes set
        --------------
        anchor_items : pandas.Series
            Item locations shifted onto the anchor scale.
        anchor_item_names : pandas.Index
            Names of the items supplied as anchors.
        anchor_adj : float
            The translation constant actually applied.
        anchor_selection : pandas.DataFrame or None
            Per-item diagnostics from whichever selection method was used
            — Anchor, Observed, Deviation, Robust z, Selected
            ('robust_z'), or Anchor, Observed, SE, Deviation, z, p,
            Selected ('wald'). None if selection_method='none' or adj was
            supplied directly.
        anchor_selected_items, anchor_dropped_items : pandas.Index or None
            Items retained / excluded as outliers. None if
            selection_method='none' or adj was supplied directly.
        anchor_original_corr, anchor_original_sd_ratio : float or None
            Correlation / SD ratio before trimming. None if
            selection_method='none' or adj was supplied directly.
        anchor_corr, anchor_sd_ratio : float or None
            Correlation / SD ratio after trimming. None if
            selection_method='none' or adj was supplied directly.
        anchor_summary : pandas.Series
            One-line summary: anchors supplied/common/selected/dropped,
            correlation and SD ratio before/after trimming (NaN if not
            computed), and the translation constant applied.
        anchor_plot : matplotlib.figure.Figure or None
            The figure from the auto-triggered plot_anchor_selection()
            call. None if plot=False, selection_method='none', or adj was
            supplied directly (no selection table to plot in those cases).
        anchor_threshold_test : pandas.Series or None
            Omnibus LR test: LL_estimated, LL_given, LR, df, p, and a
            Flagged bool. None if anchor_thresholds was not supplied.
        anchor_category_width_test : pandas.DataFrame or None
            Per-category Wald test, indexed by category: Estimate (this
            dataset's own bootstrap category width), Given (anchor_thresholds'
            width at that category), Difference, SE (from cat_width_se), z,
            p, p_corrected (if correction is set), and a Flagged bool. None
            if anchor_thresholds was not supplied.
        calibrate_anchor_runs : dict
            Every call's anchor_items/anchor_adj/anchor_summary/etc.,
            keyed by tuple(sorted(anchors.items())), so results from an
            earlier anchors call survive a later call with a different
            anchor set instead of being overwritten. E.g.
            rsm.calibrate_anchor_runs[tuple(sorted(anchors_1.items()))].anchor_summary.
        """
        if overwrite_anchors not in ("none", "rejected", "all"):
            raise ValueError("overwrite_anchors must be 'none', 'rejected', or 'all'")
        if selection_method not in ("robust_z", "wald", "none"):
            raise ValueError("selection_method must be 'robust_z', 'wald', or 'none'")

        if not isinstance(anchors, pd.Series):
            anchors = pd.Series(anchors)

        if calibrate or not hasattr(self, "items"):
            self.calibrate(
                constant=constant,
                threshold_constant=threshold_constant,
                method=method,
                matrix_power=matrix_power,
                log_lik_tol=log_lik_tol,
            )

        n_common = len(anchors.index.intersection(self.items.index))

        if adj is not None:
            tc = adj
            self.anchor_selection = None
            self.anchor_selected_items = None
            self.anchor_dropped_items = None
            self.anchor_original_corr = None
            self.anchor_original_sd_ratio = None
            self.anchor_corr = None
            self.anchor_sd_ratio = None
        elif selection_method == "wald":
            if not hasattr(self, "item_se"):
                self.std_errors(
                    no_of_samples=no_of_samples,
                    constant=constant,
                    threshold_constant=threshold_constant,
                    method=method,
                    matrix_power=matrix_power,
                    log_lik_tol=log_lik_tol,
                    seed=seed,
                )
            result = self._wald_anchor_selection(
                anchors,
                self.items,
                self.item_se,
                alpha=wald_alpha,
                min_anchors=min_anchors,
            )
            tc = result["tc"]
            self.anchor_selection = result["table"]
            self.anchor_selected_items = result["selected_anchors"]
            self.anchor_dropped_items = result["dropped_anchors"]
            self.anchor_original_corr = result["original_anchor_corr"]
            self.anchor_original_sd_ratio = result["original_anchor_sd_ratio"]
            self.anchor_corr = result["anchor_corr"]
            self.anchor_sd_ratio = result["anchor_sd_ratio"]
        elif selection_method == "robust_z":
            result = self._robust_anchor_selection(
                anchors,
                self.items,
                corr_tol=corr_tol,
                sd_ratio_tol=sd_ratio_tol,
                min_anchors=min_anchors,
            )
            tc = result["tc"]
            self.anchor_selection = result["table"]
            self.anchor_selected_items = result["selected_anchors"]
            self.anchor_dropped_items = result["dropped_anchors"]
            self.anchor_original_corr = result["original_anchor_corr"]
            self.anchor_original_sd_ratio = result["original_anchor_sd_ratio"]
            self.anchor_corr = result["anchor_corr"]
            self.anchor_sd_ratio = result["anchor_sd_ratio"]
        else:
            common = anchors.index.intersection(self.items.index)
            if len(common) == 0:
                raise ValueError(
                    "No items are common to both anchors and this dataset's "
                    "items — cannot compute a translation constant."
                )
            tc = anchors.loc[common].mean() - self.items.loc[common].mean()
            self.anchor_selection = None
            self.anchor_selected_items = None
            self.anchor_dropped_items = None
            self.anchor_original_corr = None
            self.anchor_original_sd_ratio = None
            self.anchor_corr = None
            self.anchor_sd_ratio = None

        self.anchor_adj = tc
        self.anchor_item_names = anchors.index
        self.anchor_items = self.items + tc

        common_anchor_items = anchors.index.intersection(self.items.index)

        if overwrite_anchors == "all":
            keep_given = pd.Index([])
        elif overwrite_anchors == "rejected":
            if self.anchor_selected_items is not None:
                keep_given = common_anchor_items.intersection(
                    self.anchor_selected_items
                )
            else:
                warnings.warn(
                    "overwrite_anchors='rejected' has no selected/rejected "
                    "distinction to act on here (selection_method='none' or "
                    "adj was supplied directly). Falling back to keeping all "
                    "anchor items at their supplied value, as with "
                    "overwrite_anchors='none'.",
                    UserWarning,
                    stacklevel=2,
                )
                keep_given = common_anchor_items
        else:
            keep_given = common_anchor_items

        self.anchor_items.loc[keep_given] = anchors.loc[keep_given]

        n_selected = (
            len(self.anchor_selected_items)
            if self.anchor_selected_items is not None
            else n_common
        )
        n_dropped = (
            len(self.anchor_dropped_items)
            if self.anchor_dropped_items is not None
            else 0
        )
        self.anchor_summary = pd.Series(
            {
                "Anchors supplied": len(anchors),
                "Anchors common": n_common,
                "Anchors selected": n_selected,
                "Anchors dropped": n_dropped,
                "Original corr": self.anchor_original_corr,
                "Original SD ratio": self.anchor_original_sd_ratio,
                "Final corr": self.anchor_corr,
                "Final SD ratio": self.anchor_sd_ratio,
                "Translation constant": tc,
            },
            name="Anchor calibration",
        )

        if plot and self.anchor_selection is not None:
            self.anchor_plot = self.plot_anchor_selection(
                self.anchor_selection, **(plot_kwargs or {})
            )
        else:
            self.anchor_plot = None

        if anchor_thresholds is not None:
            if not isinstance(anchor_thresholds, pd.Series):
                anchor_thresholds = pd.Series(
                    anchor_thresholds, index=range(1, self.max_score + 1)
                )

            pe_kw = dict(
                warm_corr=warm_corr,
                tolerance=tolerance,
                max_iters=max_iters,
                ext_score_adjustment=ext_score_adjustment,
            )
            ll_estimated = self._threshold_structure_ll(self.thresholds, **pe_kw)
            ll_given = self._threshold_structure_ll(anchor_thresholds, **pe_kw)
            df = self.max_score - 1

            # PAIR/CPAT are approximate (not exact MLE), so a small negative
            # LR from sampling noise doesn't necessarily mean anything —
            # floor at 0, same as andersen_lr_test does for the same reason.
            # AIC/BIC don't apply here: both candidates are point-values in
            # the exact same threshold-vector-length model, so there's no
            # real difference in parameter count to penalise for — only the
            # LR test's fixed-vs-free chi-squared framing is appropriate.
            lr = max(0.0, 2 * (ll_estimated - ll_given))
            p = float(chi2.sf(lr, df))
            flagged = p < alpha

            self.anchor_threshold_test = pd.Series(
                {
                    "LL_estimated": ll_estimated,
                    "LL_given": ll_given,
                    "LR": lr,
                    "df": df,
                    "p": p,
                    "Flagged": flagged,
                },
                name="Anchor threshold structure test",
            )

            if flagged:
                warnings.warn(
                    "anchor_thresholds differs substantially from this "
                    "dataset's own estimated threshold structure "
                    "(LR test; see anchor_threshold_test). "
                    "The rating-scale category structure may not actually "
                    "be shared with the anchor source — translation-only "
                    "equating via anchor_items assumes it is.",
                    UserWarning,
                    stacklevel=2,
                )

            # Per-category test, alongside the omnibus one: compares category
            # WIDTHS (consecutive threshold differences), not raw threshold
            # locations, so a genuine difference in one category's shape
            # doesn't cascade into a false signal at every category after it
            # (the same reasoning as dif_test's category-width/DCF test).
            if not hasattr(self, "cat_width_se"):
                self.std_errors(
                    no_of_samples=no_of_samples,
                    constant=constant,
                    threshold_constant=threshold_constant,
                    method=method,
                    matrix_power=matrix_power,
                    log_lik_tol=log_lik_tol,
                    seed=seed,
                )

            widths_estimated = self.thresholds.diff().dropna()
            widths_estimated.index = range(1, self.max_score)
            widths_given = anchor_thresholds.diff().dropna()
            widths_given.index = range(1, self.max_score)

            width_diff = widths_estimated - widths_given
            z = width_diff / self.cat_width_se
            p_width = 2 * (1 - norm.cdf(np.abs(z)))

            table = pd.DataFrame(
                {
                    "Estimate": widths_estimated,
                    "Given": widths_given,
                    "Difference": width_diff,
                    "SE": self.cat_width_se,
                    "z": z,
                    "p": p_width,
                }
            )
            table.index.name = "Category"

            if correction == "bh":
                table["p_corrected"] = self._bh_correction(table["p"])
            elif correction == "bonferroni":
                table["p_corrected"] = (table["p"] * len(table)).clip(upper=1)
            p_col = "p_corrected" if correction else "p"
            table["Flagged"] = table[p_col] < alpha
            self.anchor_category_width_test = table

            if table["Flagged"].any():
                warnings.warn(
                    "anchor_thresholds differs substantially from this "
                    "dataset's own estimated category widths at "
                    f"{list(table.index[table['Flagged']])} (per-category "
                    "Wald test; see anchor_category_width_test). The "
                    "rating-scale category structure may not actually be "
                    "shared with the anchor source at these categories — "
                    "translation-only equating via anchor_items assumes "
                    "it is.",
                    UserWarning,
                    stacklevel=2,
                )
        else:
            self.anchor_threshold_test = None
            self.anchor_category_width_test = None

        # Snapshot this run keyed by the anchors supplied, so results from an
        # earlier calibrate_anchor() call survive a later call with a
        # different anchor set instead of being overwritten in place.
        from types import SimpleNamespace
        if not hasattr(self, "calibrate_anchor_runs"):
            self.calibrate_anchor_runs = {}
        key = tuple(sorted(anchors.items()))
        self.calibrate_anchor_runs[key] = SimpleNamespace(
            anchor_items=self.anchor_items,
            anchor_item_names=self.anchor_item_names,
            anchor_adj=self.anchor_adj,
            anchor_selection=self.anchor_selection,
            anchor_selected_items=self.anchor_selected_items,
            anchor_dropped_items=self.anchor_dropped_items,
            anchor_original_corr=self.anchor_original_corr,
            anchor_original_sd_ratio=self.anchor_original_sd_ratio,
            anchor_corr=self.anchor_corr,
            anchor_sd_ratio=self.anchor_sd_ratio,
            anchor_summary=self.anchor_summary,
            anchor_plot=self.anchor_plot,
            anchor_threshold_test=self.anchor_threshold_test,
            anchor_category_width_test=self.anchor_category_width_test,
        )

    def _threshold_structure_ll(
        self, thresholds, warm_corr=True, tolerance=0.00001, max_iters=100,
        ext_score_adjustment=0.5,
    ):
        """
        Log-likelihood of this dataset's responses using self.anchor_items
        (held fixed) combined with a candidate threshold vector, with
        person locations re-estimated under that specific (items, thresholds) pair.

        Used by calibrate_anchor's anchor_thresholds diagnostic to compare
        two candidate threshold structures on a fair footing — person locations
        can't be reused across candidates since they depend on the
        threshold vector too. Uses a scratch RSM instance (same convention
        as std_errors' bootstrap and andersen_lr_test's group refits) so
        self's own calibration state is never touched.
        """
        if not isinstance(thresholds, pd.Series):
            thresholds = pd.Series(thresholds, index=range(1, self.max_score + 1))
        probe = RSM(self.responses, max_score=self.max_score)
        probe.items = self.anchor_items
        probe.thresholds = thresholds
        probe.person_estimates(
            warm_corr=warm_corr,
            tolerance=tolerance,
            max_iters=max_iters,
            ext_score_adjustment=ext_score_adjustment,
        )
        return probe._log_likelihood()

    # ------------------------------------------------------------------
    # Standard errors (bootstrap)
    # ------------------------------------------------------------------

    def std_errors(
        self,
        interval=None,
        no_of_samples=500,
        constant=None,
        threshold_constant=None,
        method="log-lik",
        matrix_power=None,
        log_lik_tol=0.000001,
        seed=None,
    ):
        """
        Estimate bootstrap standard errors for item locations and thresholds.

        Draws no_of_samples bootstrap resamples of person-level response data,
        calibrates each, and computes SDs of item location and threshold
        estimates across samples. Also computes category width SEs.

        Parameters
        ----------
        interval : float or None, default None
            CI width (e.g. 0.95). If None, only SEs computed.
        no_of_samples : int, default 500
            Number of bootstrap resamples.
        constant : float or None, default None
            Smoothing constant for bootstrap calibrations, passed to
            calibrate(); None resolves to 0.1.
        threshold_constant : float or None, default None
            Passed to calibrate() as the Stage 2 CPAT constant; None
            follows `constant` (0.03 when constant is None). See calibrate().
        method : str, default 'log-lik'
            Priority vector extraction method.
        matrix_power : int or None, default None
            Matrix power for bootstrap calibrations. None resolves to 0 for
            method='log-lik' (no powering), else 5.
        log_lik_tol : float, default 0.000001
            Convergence tolerance.
        seed : int or None, default None
            Seed for the bootstrap resampling RNG. Pass an int for fully
            reproducible standard errors; None (default) draws fresh entropy.

        Attributes set
        --------------
        item_se : pandas.Series
            Bootstrap SE for each item location.
        threshold_se : numpy.ndarray
            Bootstrap SE for each threshold (length max_score).
        cat_width_se : pandas.Series
            Bootstrap SE for each category width (threshold spacing).
        item_low / item_high : pandas.Series or None
            Bootstrap CI bounds for item locations.
        threshold_low / threshold_high : numpy.ndarray or None
            Bootstrap CI bounds for thresholds.
        item_bootstrap : pandas.DataFrame
            Bootstrap item location estimates, shape (no_of_samples, items).
        threshold_bootstrap : pandas.DataFrame
            Bootstrap threshold estimates, shape (no_of_samples, max_score).
        cat_width_bootstrap : pandas.DataFrame
            Bootstrap category width estimates.
        """
        rng = np.random.default_rng(seed)
        samples = [
            RSM(self.responses.sample(frac=1, replace=True, random_state=rng), max_score=self.max_score)
            for _ in range(no_of_samples)
        ]

        for sample in samples:
            sample.calibrate(
                constant=constant,
                threshold_constant=threshold_constant,
                method=method,
                matrix_power=matrix_power,
                log_lik_tol=log_lik_tol,
            )

        item_ests = np.array([s.items.values for s in samples])  # (B, I)
        threshold_ests = np.array([s.thresholds.values for s in samples])  # (B, K)

        sample_idx = [f"Sample {i + 1}" for i in range(no_of_samples)]

        self.item_bootstrap = pd.DataFrame(
            item_ests, index=sample_idx, columns=self.responses.columns
        )
        self.item_se = pd.Series(
            np.nanstd(item_ests, axis=0), index=self.responses.columns
        )

        self.threshold_bootstrap = pd.DataFrame(
            threshold_ests, index=sample_idx, columns=range(self.max_score)
        )
        self.threshold_se = np.nanstd(threshold_ests, axis=0)

        # Category width bootstrap: width_k = tau_{k+1} - tau_k
        cat_widths = {
            cat + 1: threshold_ests[:, cat + 1] - threshold_ests[:, cat]
            for cat in range(self.max_score - 1)
        }
        self.cat_width_bootstrap = pd.DataFrame(cat_widths, index=sample_idx)
        self.cat_width_bootstrap.columns = range(1, self.max_score)
        self.cat_width_se = {cat: np.nanstd(est) for cat, est in cat_widths.items()}
        self.cat_width_se = pd.Series(self.cat_width_se)

        if interval is not None:
            lo, hi = 50 * (1 - interval), 50 * (1 + interval)
            self.item_low = pd.Series(
                np.percentile(item_ests, lo, axis=0), index=self.responses.columns
            )
            self.item_high = pd.Series(
                np.percentile(item_ests, hi, axis=0), index=self.responses.columns
            )
            self.threshold_low = np.percentile(threshold_ests, lo, axis=0)
            self.threshold_high = np.percentile(threshold_ests, hi, axis=0)
            self.cat_width_low = {
                cat: np.percentile(est, lo) for cat, est in cat_widths.items()
            }
            self.cat_width_high = {
                cat: np.percentile(est, hi) for cat, est in cat_widths.items()
            }
        else:
            self.item_low = self.item_high = None
            self.threshold_low = self.threshold_high = None
            self.cat_width_low = self.cat_width_high = None

    # ------------------------------------------------------------------
    # Person location estimation
    # ------------------------------------------------------------------

    def person(
        self,
        persons,
        items=None,
        warm_corr=True,
        tolerance=0.00001,
        max_iters=100,
        ext_score_adjustment=0.5,
        missing_as_incorrect=False,
    ):
        """
        Estimate person locations using Newton-Raphson maximum likelihood.

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
            Person location estimates indexed by person identifier, in logits.
        """
        if isinstance(persons, str):
            persons = self.person_names if persons == "all" else [persons]

        if items is None:
            items = list(self.item_names)
        elif isinstance(items, str):
            items = list(self.item_names) if items == "all" else [items]

        item_locations = self.items.loc[items]
        person_data = self.responses.loc[persons, items]

        if missing_as_incorrect:
            person_data = person_data.fillna(0)

        person_filter = person_data.notna().astype(float)

        scores = person_data.sum(axis=1).astype(float)
        ext_scores = person_filter.sum(axis=1) * self.max_score

        # Adjust extreme scores to keep log() finite
        scores[scores == 0] = ext_score_adjustment
        scores[scores == ext_scores] -= ext_score_adjustment

        mean_item_location = item_locations.mean()

        try:
            estimates = np.log(scores) - np.log(ext_scores - scores) + mean_item_location

            # Per-person convergence mask — freeze persons once change < tolerance.
            # Without this, the log-sum-exp implementation (which gives numerically
            # valid probs for all person location values) keeps updating slowly-converging
            # persons every iteration, allowing drift of ±1 logit per step.
            active = pd.Series(True, index=list(persons))
            iters = 0

            item_location_arr = item_locations.values  # (I,)

            while active.any() and iters <= max_iters:
                active_idx = active[active].index

                probs, cats_arr = self._cat_probs_matrix(
                    estimates.loc[active_idx].values, item_location_arr, self.thresholds
                )
                # probs: (K+1, N_active, I)
                exp_score = (cats_arr[:, None, None] * probs).sum(
                    axis=0
                )  # (N_active, I)
                exp_df = pd.DataFrame(exp_score, index=active_idx, columns=items)
                exp_df *= person_filter.loc[active_idx]

                dev = (
                    cats_arr[:, None, None] - exp_score[None, :, :]
                )  # (K+1, N_active, I)
                info = (dev**2 * probs).sum(axis=0)  # (N_active, I)
                info_df = pd.DataFrame(info, index=active_idx, columns=items)
                info_df *= person_filter.loc[active_idx]

                result_list = exp_df.sum(axis=1)
                info_list = info_df.sum(axis=1)

                changes = ((result_list - scores.loc[active_idx]) / info_list).clip(
                    -1, 1
                )
                estimates.loc[active_idx] -= changes

                active.loc[active_idx] = abs(changes) > tolerance
                iters += 1

            if iters >= max_iters and active.any():
                n_nc = int(active.sum())
                warnings.warn(
                    f"{n_nc} person(s) did not converge in estimate() and will be set to NaN. "
                    f"Consider increasing max_iters or checking for degenerate response patterns.",
                    UserWarning,
                    stacklevel=2,
                )
                estimates[active] = np.nan

            if warm_corr:
                valid = estimates.notna()
                if valid.any():
                    estimates[valid] += self.warm(
                        estimates[valid],
                        items,
                        person_filter.loc[estimates.index[valid]],
                    )

        except Exception as e:
            warnings.warn(
                f"estimate() failed with exception: {e}. "
                "Returning NaN for all persons.",
                UserWarning,
                stacklevel=2,
            )
            estimates = pd.Series(np.nan, index=list(persons))

        return estimates

    def person_estimates(
        self,
        items=None,
        warm_corr=True,
        tolerance=0.00001,
        max_iters=100,
        ext_score_adjustment=0.5,
        missing_as_incorrect=False,
    ):
        """
        Estimate person locations for all persons and store as an attribute.

        Wrapper around estimate() that estimates person locations for all persons
        and stores the result as self.persons.

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
        missing_as_incorrect : bool, default False
            If True, treats missing responses as score 0 rather than
            excluding them from the likelihood. Relevant for educational
            testing contexts where non-response implies incorrect.

        Attributes set
        --------------
        persons : pandas.Series
            Person location estimates for all persons, in logits.
        """
        self.persons = self.person(
            self.person_names,
            items=items,
            warm_corr=warm_corr,
            tolerance=tolerance,
            max_iters=max_iters,
            ext_score_adjustment=ext_score_adjustment,
            missing_as_incorrect=missing_as_incorrect,
        )

    def score_lookup(
        self,
        score,
        items=None,
        warm_corr=True,
        tolerance=0.00001,
        max_iters=100,
        ext_score_adjustment=0.5,
    ):
        """
        Convert a raw total score to a person location estimate via Newton-Raphson ML.

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
            Person location estimate in logits.
        """
        if items is None or (isinstance(items, str) and items == "all"):
            items = list(self.item_names)
        elif isinstance(items, str):
            items = [items]

        item_locations = self.items.loc[items]
        ext_score = len(items) * self.max_score
        mean_item_location = item_locations.mean()

        used_score = float(score)
        if used_score == 0:
            used_score = ext_score_adjustment
        elif used_score == ext_score:
            used_score -= ext_score_adjustment

        estimate = log(used_score) - log(ext_score - used_score) + mean_item_location
        change, iters = 1.0, 0

        while abs(change) > tolerance and iters <= max_iters:
            result = sum(
                self.exp_score(estimate, diff, self.thresholds) for diff in item_locations
            )
            info = sum(
                self.variance(estimate, diff, self.thresholds) for diff in item_locations
            )
            change = max(-1.0, min(1.0, (result - used_score) / info))
            estimate -= change
            iters += 1

        if warm_corr:
            pf = pd.DataFrame(1.0, columns=items, index=[score])
            estimate += float(
                self.warm(pd.Series({score: estimate}), items, pf).iloc[0]
            )

        if iters >= max_iters:
            warnings.warn(
                "Maximum iterations reached before convergence in score_lookup(). "
                "Returned estimate may be inaccurate.",
                UserWarning,
                stacklevel=2,
            )

        return estimate

    def score_lookup_table(
        self,
        items=None,
        ext_scores=True,
        warm_corr=True,
        tolerance=0.00001,
        max_iters=100,
        ext_score_adjustment=0.5,
    ):
        """
        Build a score-to-location lookup table for all possible raw scores.

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
        person_table : pandas.Series
            Person location estimate for each possible raw score, indexed by score.
        """
        if isinstance(items, str) and items in ("all", "none"):
            items = None
        elif isinstance(items, str):
            items = [items]
        if items is None:
            items = list(self.item_names)

        no_of_items = len(items)
        item_locations = self.items.loc[items]
        total_max = no_of_items * self.max_score

        if ext_scores:
            scores = np.arange(total_max + 1)
            used_scores = scores.astype(float)
            used_scores[0] += ext_score_adjustment
            used_scores[-1] -= ext_score_adjustment
        else:
            scores = np.arange(1, total_max)
            used_scores = scores.astype(float)

        mean_item_location = item_locations.mean()
        estimates = pd.Series(
            np.log(used_scores) - np.log(total_max - used_scores) + mean_item_location,
            index=scores,
        )

        changes = pd.Series(1.0, index=scores)
        iters = 0
        item_location_arr = item_locations.values

        while abs(changes).max() > tolerance and iters <= max_iters:
            probs, cats_arr = self._cat_probs_matrix(
                estimates.values, item_location_arr, self.thresholds
            )
            exp_score = (cats_arr[:, None, None] * probs).sum(axis=0)
            exp_df = pd.DataFrame(exp_score, index=scores, columns=items)

            dev = cats_arr[:, None, None] - exp_score[None, :, :]
            info = (dev**2 * probs).sum(axis=0)
            info_df = pd.DataFrame(info, index=scores, columns=items)

            changes = ((exp_df.sum(axis=1) - used_scores) / info_df.sum(axis=1)).clip(
                -1, 1
            )
            estimates -= changes
            iters += 1

        if warm_corr:
            pf = pd.DataFrame(1.0, columns=items, index=scores)
            estimates += self.warm(estimates, items, pf)

        self.score_table = estimates

    def warm(self, person_locations, items, person_filter):
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
        person_locations : pandas.Series
            Current person location estimates, indexed by person.
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

        item_locations = self.items.loc[items]
        pf = person_filter.values if isinstance(person_filter, pd.DataFrame) else None

        probs, cats_arr = self._cat_probs_matrix(
            person_locations.values, item_locations.values, self.thresholds
        )
        # probs: (K+1, N, I)

        exp_score = (cats_arr[:, None, None] * probs).sum(axis=0)  # (N, I)
        if pf is not None:
            exp_score *= pf

        dev = cats_arr[:, None, None] - exp_score[None, :, :]  # (K+1, N, I)
        info = (dev**2 * probs).sum(axis=0)  # (N, I)
        if pf is not None:
            info *= pf

        # part_1: Σ_i Σ_k k^3 P(X_i=k) -- must use MASKED probs
        # so unobserved items contribute 0, matching exp_score and info.
        cats3 = (cats_arr**3)[:, None, None]
        masked_probs = probs * pf[None, :, :] if pf is not None else probs
        part_1 = (cats3 * masked_probs).sum(axis=0).sum(axis=1)  # (N,)

        exp_sq = exp_score**2
        part_2 = 3 * ((info + exp_sq) * exp_score).sum(axis=1)  # (N,)
        part_3 = 2 * (exp_score**3).sum(axis=1)  # (N,)

        info_sum = info.sum(axis=1)  # (N,)
        den = 2 * info_sum**2

        warm_correction = (part_1 - part_2 + part_3) / den
        return pd.Series(warm_correction, index=person_locations.index)

    def csem(self, persons=None, person_locations=None, items=None):
        """
        Compute the conditional standard error of measurement.

        CSEM = 1 / sqrt(I) where I is total Fisher information summed across
        observed items. Uses vectorised _cat_probs_matrix.

        Parameters
        ----------
        persons : list, str, or None, default None
            Person identifiers. If provided, overrides person_locations.
        person_locations : pandas.Series, float, list, numpy.ndarray, or None, default None
            Person location estimates. If None, uses self.persons, calling
            self.person_estimates() automatically to generate it if not
            already present. A raw float/list/array of locations (or any
            locations not indexed by a real person) is treated as
            hypothetical: since there is no observed response row to
            consult, all items in items are treated as answered.
        items : str, list, or None, default None
            Item subset. None uses all items.

        Returns
        -------
        pandas.Series
            CSEM values for each person/location, in logits.
        """
        # BUG FIX (original): when both persons and a custom person_locations
        # were supplied, persons always overrode person_locations by looking
        # itself up in self.persons, silently discarding the caller's table
        # (e.g. a raw-score lookup via score_table). persons now keys into
        # whichever table is in play: the supplied person_locations if one
        # was given, else self.persons — matching SLM's single-argument
        # person/person_locations behaviour.
        person_locations_supplied = person_locations is not None

        if person_locations is None:
            if not hasattr(self, "persons"):
                self.person_estimates()

            person_locations = self.persons
        if isinstance(person_locations, (int, float)):
            person_locations = pd.Series({"Location": float(person_locations)})
        if isinstance(person_locations, (list, np.ndarray)):
            person_locations = pd.Series({f"Location {a}": a for a in person_locations})
        if persons is not None:
            if not isinstance(persons, (list, pd.Index, np.ndarray)):
                persons = [persons]
            if person_locations_supplied:
                person_locations = person_locations.loc[persons]
            else:
                person_locations = self.persons.loc[persons]

        if items is None or (isinstance(items, str) and items == "all"):
            items = list(self.item_names)
        elif isinstance(items, str):
            items = [items]

        persons = person_locations.index
        item_locations = self.items.loc[items]
        # BUG FIX (original): unconditionally indexed self.responses by persons,
        # which failed for hypothetical locations (raw floats/lists, or any
        # person_locations index not matching self.responses) with no matching
        # row. Real persons are still filtered by their actual missing-response
        # pattern; hypothetical locations are treated as fully answered.
        is_real_person = persons.isin(self.responses.index)
        person_filter = self.responses.reindex(persons)[items].notna().astype(float)
        person_filter.loc[~is_real_person] = 1.0

        probs, cats_arr = self._cat_probs_matrix(
            person_locations.values, item_locations.values, self.thresholds
        )
        exp_score = (cats_arr[:, None, None] * probs).sum(axis=0)
        pf = person_filter.values
        exp_score *= pf

        dev = cats_arr[:, None, None] - exp_score[None, :, :]
        info = (dev**2 * probs).sum(axis=0) * pf

        return pd.Series(1.0 / (info.sum(axis=1) ** 0.5), index=persons)

    # ------------------------------------------------------------------
    # Descriptive / count methods
    # ------------------------------------------------------------------

    def category_counts_df(self, persons=None, items=None, counts_name=None):
        """
        Build a response frequency table for one or more persons, across
        one or more items.

        All items share the same max_score in RSM, so there are no blank
        cells. Computes category counts (0 through max_score), total
        valid responses, and missing responses, over the requested
        persons. Appends a Total row.

        Parameters
        ----------
        persons : str, list, or None, default None
            Person(s) to include (a single person name or a list of
            names) -- e.g. to compare a score-split or exogenous-variable
            group against the rest. None uses all persons.
        items : str, list, or None, default None
            Item(s) to include (as elsewhere in the package, a single item
            name or a list of item names). None uses all items.
        counts_name : str, int, or None, default None
            If None, stores the result as self.category_counts_table
            (overwriting any previous call). If given, stores it instead
            under that key in self.counts (created if it doesn't already
            exist) -- e.g. self.counts['group_a'] and
            self.counts.group_a (dot access works when counts_name is a
            valid Python identifier) -- so a succession of tables (e.g.
            one per exogenous-variable group) can be kept side by side
            for comparison rather than overwriting each other.

        Returns
        -------
        pandas.DataFrame
            Items as rows, categories plus Total and Missing as columns.
            A Total row is appended. All values are integers.
        """
        if items is None:
            items = list(self.responses.columns)
        elif isinstance(items, str):
            items = [items]

        if persons is None:
            persons = list(self.responses.index)
        elif isinstance(persons, str):
            persons = [persons]

        subset = self.responses.loc[persons, items]

        cat_counts = {
            item: subset[item]
            .value_counts()
            .reindex(range(self.max_score + 1), fill_value=0)
            .astype(int)
            for item in items
        }
        df = pd.DataFrame(cat_counts).T.sort_index(axis=1)
        df["Total"] = subset.count()
        df["Missing"] = len(persons) - df["Total"]
        df.loc["Total"] = df.sum()
        df = df.astype(int)

        if counts_name is None:
            self.category_counts_table = df
        else:
            if not hasattr(self, "counts"):
                from raschpy.base import _Namespace

                self.counts = _Namespace()
            self.counts[counts_name] = df

        return df

    # ------------------------------------------------------------------
    # Fit statistics
    # ------------------------------------------------------------------

    def _log_likelihood(self, responses=None, persons=None):
        if responses is None:
            responses = self.responses
        if persons is None:
            persons = self.persons
        scores = responses.sum(axis=1)
        max_scores = responses.notna().sum(axis=1) * self.max_score
        non_extreme = responses.index[(scores > 0) & (scores < max_scores)]
        persons = persons.reindex(non_extreme).dropna()
        persons = persons[persons.abs() <= 20]
        obs_arr = responses.loc[persons.index].values
        valid = ~np.isnan(obs_arr)
        probs, _ = self._cat_probs_matrix(
            persons.values, self.items.values, self.thresholds
        )
        obs_int = np.where(valid, obs_arr, 0).astype(int)
        n_idx, i_idx = np.meshgrid(
            np.arange(obs_arr.shape[0]), np.arange(obs_arr.shape[1]), indexing="ij"
        )
        prob_obs = probs[obs_int, n_idx, i_idx]
        prob_obs[~valid] = np.nan
        return float(np.nansum(np.log(prob_obs)))

    def fit_statistics(
        self,
        warm_corr=True,
        se=True,
        test_stats=True,
        trim_cat_prob_dict=False,
        tolerance=0.00001,
        max_iters=100,
        ext_score_adjustment=0.5,
        method="log-lik",
        constant=None,
        threshold_constant=None,
        matrix_power=None,
        no_of_samples=500,
        log_lik_tol=0.000001,
        interval=None,
        seed=None,
    ):
        """
        Compute all item, threshold, person, and test-level fit statistics.

        Auto-triggers calibrate(), std_errors(), and person_estimates() if not yet
        run. Uses vectorised _cat_probs_matrix. Applies a cell-level guard
        (p > 0.9999) to prevent kurtosis/info^2 overflow in outfit statistics.
        Threshold fit statistics are computed by dichotomising at each threshold.

        Parameters
        ----------
        warm_corr : bool, default True
            Warm bias correction for person location estimates.
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
        method : str, default 'log-lik'
            Priority vector extraction method.
        constant : float or None, default None
            Smoothing constant, passed to calibrate(); None resolves to 0.1.
        threshold_constant : float or None, default None
            Passed to calibrate() as the Stage 2 CPAT constant; None
            follows `constant` (0.03 when constant is None). See calibrate().
        matrix_power : int or None, default None
            Matrix power for calibration. None resolves to 0 for
            method='log-lik' (no powering), else 5.
        no_of_samples : int, default 500
            Bootstrap samples.
        log_lik_tol : float, default 0.000001
            Convergence tolerance for calibration.
        interval : float or None, default None
            CI width for bootstrap estimates.
        seed : int or None, default None
            Seed passed through to the internal std_errors() call (only
            used if item SEs aren't already computed). None draws fresh
            entropy each call.

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

        if not hasattr(self, "thresholds"):
            self.calibrate(
                constant=constant,
                threshold_constant=threshold_constant,
                method=method,
                matrix_power=matrix_power,
                log_lik_tol=log_lik_tol,
            )
        if se and not hasattr(self, "threshold_se"):
            self.std_errors(
                interval=interval,
                no_of_samples=no_of_samples,
                constant=constant,
                threshold_constant=threshold_constant,
                method=method,
                seed=seed,
            )
        if not hasattr(self, "persons"):
            self.person_estimates(
                warm_corr=warm_corr,
                tolerance=tolerance,
                max_iters=max_iters,
                ext_score_adjustment=ext_score_adjustment,
            )
        if not se:
            test_stats = False

        # Count valid responses per item and per person (before extreme filter)
        item_count = self.responses.notna().sum(axis=0)
        person_count = self.responses.notna().sum(axis=1)

        df = self.responses.copy()
        scores = df.sum(axis=1)
        max_scores = df.notna().sum(axis=1) * self.max_score
        df = df[(scores > 0) & (scores < max_scores)]
        missing_mask = df.notna().astype(float)
        person_locations = self.persons.loc[df.index]

        # Exclude persons with extreme person location estimates (diverged NR)
        person_locations = person_locations[person_locations.abs() <= 20]
        df = df.loc[person_locations.index]
        missing_mask = missing_mask.loc[person_locations.index]

        item_location_arr = self.items.values

        probs, cats_arr = self._cat_probs_matrix(
            person_locations.values, item_location_arr, self.thresholds
        )
        # probs: (K+1, N, I)

        # Cell-level guard: exclude person-item cells where one category
        # has near-certain probability (p > 0.9999, WINSTEPS convention).
        # Prevents kurtosis/info^2 overflow in outfit q-factor calculation.
        max_cat_prob = pd.DataFrame(
            probs.max(axis=0), index=person_locations.index, columns=self.item_names
        )
        degenerate = max_cat_prob > 0.9999

        exp_score = (cats_arr[:, None, None] * probs).sum(axis=0)  # (N, I)
        self.exp_score_df = (
            pd.DataFrame(exp_score, index=person_locations.index, columns=self.item_names)
            * missing_mask
        )
        self.exp_score_df[degenerate] = np.nan

        dev = cats_arr[:, None, None] - exp_score[None, :, :]  # (K+1, N, I)
        info = (dev**2 * probs).sum(axis=0)  # (N, I)
        kurtosis = ((dev**4) * probs).sum(axis=0)  # (N, I)

        self.info_df = (
            pd.DataFrame(info, index=person_locations.index, columns=self.item_names)
            * missing_mask
        )
        self.info_df[degenerate] = np.nan

        self.kurtosis_df = (
            pd.DataFrame(kurtosis, index=person_locations.index, columns=self.item_names)
            * missing_mask
        )
        self.kurtosis_df[degenerate] = np.nan

        self.residual_df = self.responses.reindex(df.index) - self.exp_score_df
        self.std_residual_df = self.residual_df / (self.info_df**0.5)

        self.log_likelihood = self._log_likelihood()

        n_persons = int(((scores > 0) & (scores < max_scores)).sum())
        k = (self.no_of_items - 1) + (self.max_score - 1)
        self.aic = 2 * k - 2 * self.log_likelihood
        self.bic = k * np.log(n_persons) - 2 * self.log_likelihood

        self.cat_prob_dict = {
            cat: pd.DataFrame(
                probs[cat], index=person_locations.index, columns=self.item_names
            )
            for cat in range(probs.shape[0])
        }
        if trim_cat_prob_dict:
            for cat in self.cat_prob_dict:
                self.cat_prob_dict[cat] = self.cat_prob_dict[cat].loc[df.index]

        # --- Item fit ---
        self.item_outfit_ms = (self.std_residual_df**2).mean()
        self.item_infit_ms = (self.residual_df**2).sum() / self.info_df.sum()

        item_outfit_q = (
            ((self.kurtosis_df / (self.info_df**2)) / (item_count**2)).sum()
            - (1 / item_count)
        ) ** 0.5
        self.item_outfit_zstd = ((self.item_outfit_ms ** (1 / 3)) - 1) * (
            3 / item_outfit_q
        ) + (item_outfit_q / 3)

        item_infit_q = (
            (self.kurtosis_df - self.info_df**2).sum() / (self.info_df.sum() ** 2)
        ) ** 0.5
        self.item_infit_zstd = ((self.item_infit_ms ** (1 / 3)) - 1) * (
            3 / item_infit_q
        ) + (item_infit_q / 3)

        self.response_counts = self.responses.count(axis=0)
        self.item_facilities = self.responses.mean(axis=0) / self.max_score

        (self.point_measure, self.exp_point_measure) = self.pt_meas(
            self.persons, self.exp_score_df, self.info_df
        )

        # --- Threshold fit (dichotomised across all items) ---
        # For RSM, thresholds are shared so dich_thresh covers ALL items
        # for each threshold level, not per-item as in PCM.
        person_location_df = pd.DataFrame(
            np.tile(self.persons.values[:, None], (1, self.no_of_items)),
            index=self.responses.index,
            columns=self.responses.columns,
        )

        dich_thresh = {}
        dich_thresh_exp = {}
        dich_thresh_var = {}
        dich_thresh_kur = {}
        dich_residuals = {}
        dich_std_residuals = {}

        for t in range(self.max_score):
            # Dichotomise: keep only persons scoring t or t+1, recode as 0/1
            dich = self.responses.where(self.responses.isin([t, t + 1]), np.nan) - t
            dich_thresh[t + 1] = dich

            mm = dich.notna().astype(float).replace(0, np.nan)

            # Threshold location for threshold t+1: item_location_i + tau_{t+1}
            # item_location_df[n,i] = delta_i + tau_{t+1}  (identical across persons,
            # so tile the (1, I) row vector to (N, I) before constructing DataFrame)
            item_location_df = pd.DataFrame(
                np.tile(
                    self.items.values + self.thresholds[t + 1],
                    (len(self.responses.index), 1),
                ),
                index=self.responses.index,
                columns=self.responses.columns,
            )

            p = 1.0 / (1.0 + np.exp(item_location_df - person_location_df))
            p_masked = p * mm

            dich_thresh_exp[t + 1] = p_masked
            dich_thresh_var[t + 1] = p_masked * (1 - p_masked) * mm
            dich_thresh_kur[t + 1] = (
                ((-p_masked) ** 4) * (1 - p_masked) + ((1 - p_masked) ** 4) * p_masked
            ) * mm
            dich_residuals[t + 1] = dich - p_masked
            dich_std_residuals[t + 1] = dich_residuals[t + 1] / (
                dich_thresh_var[t + 1] ** 0.5
            )

        dich_thresh_count = {
            t + 1: dich_thresh[t + 1].count().sum() for t in range(self.max_score)
        }

        self.threshold_outfit_ms = pd.Series(
            {
                t
                + 1: (
                    (dich_std_residuals[t + 1] ** 2).sum().sum()
                    / dich_thresh_count[t + 1]
                    if dich_thresh_count[t + 1] > 0
                    else np.nan
                )
                for t in range(self.max_score)
            }
        )

        self.threshold_infit_ms = pd.Series(
            {
                t
                + 1: (
                    (dich_residuals[t + 1] ** 2).sum().sum()
                    / dich_thresh_var[t + 1].sum().sum()
                    if dich_thresh_var[t + 1].sum().sum() > 0
                    else np.nan
                )
                for t in range(self.max_score)
            }
        )

        threshold_outfit_q = (
            pd.Series(
                {
                    t
                    + 1: (
                        (
                            (dich_thresh_kur[t + 1] / (dich_thresh_var[t + 1] ** 2))
                            / (dich_thresh_count[t + 1] ** 2)
                        )
                        .sum()
                        .sum()
                        - (1 / dich_thresh_count[t + 1])
                        if dich_thresh_count[t + 1] > 0
                        else np.nan
                    )
                    for t in range(self.max_score)
                }
            )
            ** 0.5
        )

        self.threshold_outfit_zstd = ((self.threshold_outfit_ms ** (1 / 3)) - 1) * (
            3 / threshold_outfit_q
        ) + (threshold_outfit_q / 3)

        threshold_infit_q = (
            pd.Series(
                {
                    t
                    + 1: (
                        (dich_thresh_kur[t + 1] - dich_thresh_var[t + 1] ** 2)
                        .sum()
                        .sum()
                        / (dich_thresh_var[t + 1].sum().sum() ** 2)
                        if dich_thresh_var[t + 1].sum().sum() > 0
                        else np.nan
                    )
                    for t in range(self.max_score)
                }
            )
            ** 0.5
        )

        self.threshold_infit_zstd = ((self.threshold_infit_ms ** (1 / 3)) - 1) * (
            3 / threshold_infit_q
        ) + (threshold_infit_q / 3)

        person_location_deviation = self.persons - self.persons.mean()

        # Threshold point-measure correlations
        pm_num = pd.Series(
            {
                t
                + 1: (
                    (dich_thresh[t + 1] - dich_thresh[t + 1].mean())
                    .mul(person_location_deviation, axis=0)
                    .sum()
                    .sum()
                    if dich_thresh[t + 1].count().sum() > 0
                    else np.nan
                )
                for t in range(self.max_score)
            }
        )
        pm_den = pd.Series(
            {
                t
                + 1: (
                    ((dich_thresh[t + 1] - dich_thresh[t + 1].mean()) ** 2).sum().sum()
                    * (person_location_deviation**2).sum()
                )
                ** 0.5
                for t in range(self.max_score)
            }
        )
        self.threshold_point_measure = pm_num / pm_den

        exp_pm_dict = {
            t + 1: dich_thresh_exp[t + 1] - dich_thresh_exp[t + 1].mean()
            for t in range(self.max_score)
        }
        exp_pm_num = pd.Series(
            {
                t + 1: exp_pm_dict[t + 1].mul(person_location_deviation, axis=0).sum().sum()
                for t in range(self.max_score)
            }
        )
        exp_pm_den = pd.Series(
            {
                t + 1: ((exp_pm_dict[t + 1] ** 2) + dich_thresh_var[t + 1]).sum().sum()
                for t in range(self.max_score)
            }
        )
        exp_pm_den *= (person_location_deviation**2).sum()
        exp_pm_den = exp_pm_den**0.5
        self.threshold_exp_point_measure = exp_pm_num / exp_pm_den

        self.threshold_rmsr = pd.Series(
            {
                t
                + 1: (
                    (
                        (dich_residuals[t + 1] ** 2).sum().sum()
                        / dich_residuals[t + 1].count().sum()
                    )
                    ** 0.5
                    if dich_residuals[t + 1].count().sum() > 0
                    else np.nan
                )
                for t in range(self.max_score)
            }
        )

        # Threshold discrimination
        differences = {
            t
            + 1: pd.DataFrame(
                self.persons.values[:, None]
                - (self.items.values[None, :] + self.thresholds[t + 1]),
                index=self.responses.index,
                columns=self.responses.columns,
            )
            for t in range(self.max_score)
        }
        disc_num = pd.Series(
            {
                t + 1: (differences[t + 1] * dich_residuals[t + 1]).sum().sum()
                for t in range(self.max_score)
            }
        )
        disc_den = pd.Series(
            {
                t + 1: (dich_thresh_var[t + 1] * differences[t + 1] ** 2).sum().sum()
                for t in range(self.max_score)
            }
        )
        self.threshold_discrimination = 1 + disc_num / disc_den

        # --- Person fit ---
        self.csem_vector = 1.0 / (self.info_df.sum(axis=1) ** 0.5)
        self.rsem_vector = (
            (self.residual_df**2).sum(axis=1) ** 0.5
        ) / self.info_df.sum(axis=1)

        self.person_outfit_ms = (self.std_residual_df**2).mean(axis=1)
        self.person_outfit_ms.name = "Outfit MS"
        self.person_infit_ms = (self.residual_df**2).sum(axis=1) / self.info_df.sum(
            axis=1
        )
        self.person_infit_ms.name = "Infit MS"

        base_df = self.kurtosis_df / (self.info_df**2)
        base_df = base_df.div(person_count**2, axis=0)
        person_outfit_q = (base_df.sum(axis=1) - 1 / person_count) ** 0.5
        self.person_outfit_zstd = ((self.person_outfit_ms ** (1 / 3)) - 1) * (
            3 / person_outfit_q
        ) + (person_outfit_q / 3)
        self.person_outfit_zstd.name = "Outfit Z"

        person_infit_q = (
            (self.kurtosis_df - self.info_df**2).sum(axis=1)
            / (self.info_df.sum(axis=1) ** 2)
        ) ** 0.5
        self.person_infit_zstd = ((self.person_infit_ms ** (1 / 3)) - 1) * (
            3 / person_infit_q
        ) + (person_infit_q / 3)
        self.person_infit_zstd.name = "Infit Z"

        # --- Test-level fit ---
        if test_stats:
            self.isi = (self.items.var() / (self.item_se**2).mean() - 1) ** 0.5
            self.item_strata = (4 * self.isi + 1) / 3
            self.item_reliability = self.isi**2 / (1 + self.isi**2)

            # BUG FIX: original RSM formula was:
            #   (var^0.5 - mean_rsem2) / mean_rsem2^0.5   <- wrong: sqrt taken early
            # Correct Wright & Masters formula:
            #   sqrt((var - mean_rsem2) / mean_rsem2)
            mean_rsem2 = (self.rsem_vector**2).mean()
            self.psi = ((np.var(self.persons) - mean_rsem2) / mean_rsem2) ** 0.5
            self.person_strata = (4 * self.psi + 1) / 3
            self.person_reliability = self.psi**2 / (1 + self.psi**2)

    # ------------------------------------------------------------------
    # Residual correlation / PCA
    # ------------------------------------------------------------------

    def andersen_lr_test(
        self,
        split_by="person_location",
        covariate=None,
        warm_corr=True,
        tolerance=0.00001,
        max_iters=100,
        ext_score_adjustment=0.5,
        constant=None,
        threshold_constant=None,
        method="log-lik",
        matrix_power=None,
        log_lik_tol=0.000001,
    ):
        """
        Andersen (1973) likelihood ratio test of parameter invariance.

        split_by='person_location'/'score' (general model-fit / invariance testing)
        is DISABLED as of 2026-07-06 — see NotImplementedError raised below.
        split_by='exogenous' (DIF testing) is unaffected and remains fully
        supported.

        Splits persons into two groups, fits the model separately in each
        group, and tests whether item parameters are invariant across
        groups. Groups are formed either by a median split on person location or
        raw score, or by an exogenous person covariate (e.g. Gender) for
        differential item functioning.

        Parameters
        ----------
        split_by : str, default 'person_location'
            Split criterion: 'person_location' (ML person estimates, median split),
            'score' (raw scores, median split), or 'exogenous' (an
            external person covariate — requires `covariate` and
            self.exogenous to be set). 'exogenous' requires the covariate
            to have exactly two distinct non-null values; covariates with
            more than two levels are not supported here — use dif_test()
            instead, which supports multi-level covariates directly.
        covariate : str or None, default None
            Column name in self.exogenous to split by. Required (and
            only used) when split_by='exogenous'. Persons with a missing
            value for this covariate are excluded from both groups and
            from the full-sample comparison fit, so the LR decomposition
            stays valid.
        warm_corr : bool, default True
            Warm bias correction for person location estimates.
        tolerance, max_iters, ext_score_adjustment : floats
            Person estimation kwargs passed to group models.
        constant, method, matrix_power, log_lik_tol : floats
            Calibration kwargs passed to group models.

        Attributes set
        --------------
        andersen_lr : float
            Likelihood ratio statistic. Each group's own log-likelihood
            (the H1 side) is computed by plugging in person location estimates
            from the pooled (combined-group) model rather than the
            group's own separately-fit person locations, so the comparison
            differs from the pooled model only in item parameters —
            otherwise nuisance person location parameters are re-optimised
            independently on each side, inflating the statistic beyond
            what df accounts for.
        andersen_df : int
            Degrees of freedom (no_of_items - 1) + (max_score - 1).
        andersen_p : float
            p-value from chi-squared distribution.
        andersen_groups : dict
            {group_name: RSM} — fitted group models for inspection. Group
            names are 'low'/'high' for split_by='person_location'/'score', or the
            two observed covariate values for split_by='exogenous'.
        andersen_summary : pandas.Series
            LR statistic, df, and p-value.
        """
        from raschpy.rsm import RSM

        if split_by not in ("person_location", "score", "exogenous"):
            raise ValueError("split_by must be 'person_location', 'score', or 'exogenous'")

        if split_by in ("person_location", "score"):
            raise NotImplementedError(
                "andersen_lr_test(split_by='person_location'/'score') is disabled as a "
                "general Rasch model-fit / parameter-invariance test. A 2026-07 "
                "simulation study (varying N from 100 to 4000) found the LR "
                "statistic floors to 0 (p=1.0, 'no misfit') in 30-90% of "
                "replications depending on model and N, and the floor rate does "
                "NOT improve with more data. A follow-up power study injecting "
                "genuine item-location differences of up to 3 logits between "
                "the compared groups found the rejection rate and mean LR do not "
                "respond to the true effect size at all — the test has no "
                "demonstrated power in either direction. Root cause: PAIR/CPAT "
                "are matrix-algebraic pairwise-comparison estimators, not "
                "likelihood methods of any kind (not even pseudo-likelihood) — "
                "they do not maximise the Rasch response likelihood that "
                "_log_likelihood() evaluates afterward. A valid LR test requires "
                "both the restricted and unrestricted models to be fit by "
                "maximising the same likelihood surface, so that the "
                "unrestricted model's log-likelihood is guaranteed >= the "
                "restricted model's; PAIR/CPAT gives no such guarantee, and "
                "there is no reason to expect the gap to close with more data, "
                "since neither was ever targeting the likelihood surface. This "
                "will be re-enabled once a genuine (C)ML calibration path is "
                "available for this test. NOTE: this does NOT extend to "
                "split_by='exogenous' or to dif_test() (Wald test, per-item LR "
                "test, and omnibus LR test) — those were separately verified via "
                "simulation to have correct null calibration and power that "
                "scales cleanly with true DIF magnitude, and remain fully "
                "supported."
            )

        if not hasattr(self, "thresholds"):
            self.calibrate(
                constant=constant,
                threshold_constant=threshold_constant,
                method=method,
                matrix_power=matrix_power,
                log_lik_tol=log_lik_tol,
            )
        if not hasattr(self, "persons"):
            self.person_estimates(
                warm_corr=warm_corr,
                tolerance=tolerance,
                max_iters=max_iters,
                ext_score_adjustment=ext_score_adjustment,
            )

        scores = self.responses.sum(axis=1)
        max_scores = self.responses.notna().sum(axis=1) * self.max_score
        non_extreme = self.responses.index[(scores > 0) & (scores < max_scores)]

        group_idx = self._resolve_andersen_groups(
            split_by, covariate, non_extreme, self.persons, scores
        )

        # Full-model LL restricted to persons in either group
        combined_idx = group_idx[list(group_idx)[0]].append(group_idx[list(group_idx)[1]])

        # Re-estimate full model on combined subset so the LR comparison is fair
        m_full = RSM(self.responses.loc[combined_idx], max_score=self.max_score)
        m_full.calibrate(
            constant=constant,
            threshold_constant=threshold_constant,
            method=method,
            matrix_power=matrix_power,
            log_lik_tol=log_lik_tol,
        )
        m_full.person_estimates(
            warm_corr=warm_corr,
            tolerance=tolerance,
            max_iters=max_iters,
            ext_score_adjustment=ext_score_adjustment,
        )
        ll_full = m_full._log_likelihood()

        group_lls = {}
        group_models = {}
        for name, idx in group_idx.items():
            m = RSM(self.responses.loc[idx], max_score=self.max_score)
            m.calibrate(
                constant=constant,
                threshold_constant=threshold_constant,
                method=method,
                matrix_power=matrix_power,
                log_lik_tol=log_lik_tol,
            )
            m.person_estimates(
                warm_corr=warm_corr,
                tolerance=tolerance,
                max_iters=max_iters,
                ext_score_adjustment=ext_score_adjustment,
            )
            pooled_persons = m_full.persons.reindex(m.responses.index)
            group_lls[name] = m._log_likelihood(persons=pooled_persons)
            group_models[name] = m

        lr = -2 * (ll_full - sum(group_lls.values()))
        if lr < 0:
            warnings.warn(
                "Andersen LR statistic is negative due to PAIR estimation approximation "
                "and has been floored at 0. This indicates no evidence of misfit.",
                UserWarning,
            )
            lr = 0.0
        df = (self.no_of_items - 1) + (self.max_score - 1)

        self.andersen_lr = lr
        self.andersen_df = df
        self.andersen_p = float(chi2.sf(lr, df))
        self.andersen_groups = group_models
        self.andersen_summary = pd.Series(
            {"LR statistic": lr, "df": df, "p-value": self.andersen_p},
            name="Andersen LR test",
        )

    def dif_test(
        self,
        covariate,
        reference=None,
        selection_method="wald",
        corr_tol=0.95,
        sd_ratio_tol=1.1,
        min_anchors=6,
        wald_alpha=0.05,
        test="wald",
        omnibus=True,
        welch=False,
        size_adjust=False,
        reference_n=100,
        correction="bh",
        alpha=0.05,
        logit_threshold=0.43,
        category=False,
        category_thresholds=(0.43, 0.64),
        category_alpha=0.05,
        no_of_samples=500,
        plot=False,
        plot_kwargs=None,
        warm_corr=True,
        tolerance=0.00001,
        max_iters=100,
        ext_score_adjustment=0.5,
        constant=None,
        threshold_constant=None,
        method="log-lik",
        matrix_power=None,
        log_lik_tol=0.000001,
        seed=None,
    ):
        """
        Differential Item Functioning (DIF) test by an exogenous person
        covariate (e.g. Gender, L1).

        Splits non-extreme persons into groups by the named column in
        self.exogenous, designates one group as the reference (by default
        the largest), and compares each other ("focal") group against it
        individually — a reference-group design, not all-pairwise.

        Two independent components are tested:

        1. Item-location DIF: both groups are calibrated independently
           (each self-centred), then purified onto a common item scale
           before testing — so that genuine DIF items cannot contaminate
           the scale used to test for DIF in the first place — via either
           _wald_anchor_selection() (default) or _robust_anchor_selection().
           Item-level DIF is then tested via a per-item Wald test
           (test='wald', default), a per-item likelihood-ratio test
           (test='lr'), or both (test='both'). An omnibus LR test
           (Andersen-style: reference vs focal, all items jointly) runs
           by default (omnibus=True). The Wald test uses bootstrap
           standard errors from std_errors(), computed for both groups
           regardless of selection_method (so selection_method='wald'
           costs nothing extra here). The LR test (per-item and omnibus)
           needs person locations instead, which are otherwise never
           estimated by this method — see calibrate_anchor's/SLM's
           dif_test docstring for the exact per-item LR mechanics (H1 =
           each group's own natively-calibrated fit, computed once;
           H0_i = item i's location pooled across groups, precision-
           weighted, with that group's person locations re-estimated under the
           swap). Results in self.dif_table and self.dif_omnibus_table.

        2. Threshold-structure DIF (DCF, differential category functioning):
           RSM's threshold vector
           (self.thresholds) is shared across all items and describes
           relative category structure, not item location — it is not
           translated by the item-scale purification in (1) (same
           reasoning as calibrate_anchor's own threshold handling). Tested
           via category *widths* (thresholds[k+1] - thresholds[k]) rather
           than raw threshold locations — locations are partial sums of
           widths, so a single genuine width change cascades into every
           downstream threshold location, smearing/mis-localising the
           signal if tested directly; widths isolate it to the one
           category that actually changed. Each of the max_score-1
           category widths is Wald-tested directly, reference vs. focal,
           using cat_width_se (from std_errors() — differenced *within*
           each bootstrap resample before taking the std, so it already
           reflects Cov(threshold_k, threshold_{k+1}) without a separate
           covariance term), with its own multiple-comparison correction
           pool. Not affected by test=/omnibus= — those only govern
           component (1). Results in self.threshold_dif_table.

        Parameters
        ----------
        covariate : str
            Column name in self.exogenous to group persons by.
        reference : str or None, default None
            Value of `covariate` to use as the reference group. If None,
            the largest group is used.
        selection_method : {'wald', 'robust_z', 'none'}, default 'wald'
            Only affects the item-location component — threshold-structure
            DIF is never purified/shifted.
            'wald': _wald_anchor_selection() — per-item significance test
            (z = (reference - (focal + tc)) / SE(focal), tc precision-
            weighted, sequential exclusion) using the bootstrap SEs this
            method computes anyway. Uses wald_alpha and min_anchors.
            'robust_z': _robust_anchor_selection() — Iglewicz & Hoaglin
            modified z-score / MAD-based iterative trim (corr_tol,
            sd_ratio_tol, min_anchors) — no significance test, purely
            descriptive-statistics based.
            'none': plain mean shift over every common item, no trimming.
        corr_tol, sd_ratio_tol : floats
            Passed to _robust_anchor_selection() (selection_method=
            'robust_z' only).
        min_anchors : int, default 6
            Floor on surviving common items — passed to whichever
            selection method is active ('robust_z' or 'wald').
        wald_alpha : float, default 0.05
            Significance level for each item's Wald test during
            purification (selection_method='wald' only) — distinct from
            `alpha` below, which governs DIF flagging itself.
        test : {'wald', 'lr', 'both'}, default 'wald'
            Per-item item-location DIF test(s) to compute. 'wald' matches
            prior behaviour exactly (no added cost). 'lr'/'both'
            additionally compute the per-item likelihood-ratio test (extra
            cost: two person-location re-estimations per item per focal group).
            The existing 'Flagged' column in dif_table always reflects the
            Wald test; 'Flagged_LR' is added when test is 'lr' or 'both'.
            Does not affect threshold_dif_table.
        omnibus : bool, default True
            If True, also runs an Andersen-style omnibus LR test
            (reference vs. focal, every item jointly) per focal group for
            the item-location component, stored in dif_omnibus_table.
            Cheap relative to the per-item LR test — one extra combined-
            group model fit per focal group, not per item. The H1 side
            (ll_ref + ll_focal) plugs in person location estimates from the pooled
            reference+focal model rather than each group's own separately-
            fit person locations — otherwise the two sides of the comparison
            re-optimise nuisance person-location parameters independently, which
            inflates the LR statistic beyond what df=k-1 accounts for
            (confirmed by null-DIF simulation: ~1.3-1.7x nominal Type I
            error with own-group person locations, ~nominal with pooled person locations).
        welch : bool, default False
            If True, every Wald test here (item-location AND threshold-
            structure) becomes a Welch's t-test: the statistic is
            unchanged (diff / combined SE) but its p-value is computed
            against a t-distribution with Welch-Satterthwaite degrees of
            freedom (using each group's own person count) instead of a
            normal distribution — more conservative than a z-test when
            group sizes are small, unequal, or the two SEs differ
            substantially. Adds a 'df' column to both dif_table and
            threshold_dif_table. Does not affect the LR test or omnibus
            (already exact chi-squared tests) or _wald_anchor_selection (a
            one-sample comparison against a fixed anchor value, not the
            two-independent-samples case Welch's t-test addresses).
        size_adjust : bool, default False
            If True, rescales each group's own SE (item location AND
            threshold — same scope as welch, unlike category below) to
            what it would be at a standard reference sample size
            (Tristán, 2006, Rasch Measurement Transactions 20:3 — the
            opposite problem from welch's: at very large N, SE shrinks
            enough that trivial differences become "significant",
            swamping the logit-magnitude thresholds' intent). Each
            group's own SE is rescaled independently by
            sqrt(actual_n / reference_n) using that group's own person
            count, then combined as usual. Applied to the Wald/Welch
            test, the category tests, and (if welch=True) the Welch-
            Satterthwaite df, which then also uses reference_n rather
            than each group's actual n, for internal consistency.
        reference_n : int, default 100
            Reference sample size for size_adjust=True. 100 is Tristán's own
            default recommendation; ~60 is suggested there for closer
            alignment with the ETS Category B boundary specifically
            (100*(0.43/0.55)**2). Unused unless size_adjust=True.
        correction : 'bh', 'bonferroni', or None, default 'bh'
            Multiple-comparison correction applied across items (within
            each focal-group comparison) and, separately, across
            thresholds (within each focal-group comparison) — applied to
            the Wald and per-item LR p-values alike if both are computed.
            Not applied to the omnibus test.
        alpha : float, default 0.05
            p-value threshold for flagging (Wald, per-item LR, and
            omnibus alike). Compared against the corrected p-value where
            correction applies.
        logit_threshold : float, default 0.43
            Absolute logit-difference threshold for flagging (ETS-style
            convention), applied to the Wald test in both components. An
            item/threshold is flagged (Flagged or Flagged_LR) only if
            both this and the relevant p-value threshold are met — the
            LR-based flag reuses the same purified item-location Difference
            estimate as the Wald flag as its effect-size gate.
        category : bool, default False
            If True, adds an ETS-style 'Category' column to dif_table
            (item-location component only — the 0.43/0.64 logit defaults
            are calibrated for item-location DIF specifically, not
            threshold DIF, so this doesn't extend to threshold_dif_table):
            'A' (negligible), 'B+'/'B-' (slight to moderate), or 'C+'/'C-'
            (moderate to large), following Zwick, Thayer & Lewis (1999).
            Sign: '+' means Difference > 0 (item harder for focal, i.e.
            DIF against reference by this package's convention); '-'
            means DIF against focal. Uses two tests: 'B' requires
            |Difference| >= category_thresholds[0] AND prob(DIF=0) <
            category_alpha (the existing Wald/Welch p-value, reused); 'C'
            requires |Difference| >= category_thresholds[1] AND a
            *different*, one-sided test — prob(|DIF| <=
            category_thresholds[0]) < category_alpha, i.e. whether
            |Difference| is significantly above the B/C boundary itself,
            not just significantly nonzero. Uses the same reference
            distribution as welch (t with Satterthwaite df, or normal).
        category_thresholds : (float, float), default (0.43, 0.64)
            (B boundary, C boundary) in logits, per Zwick et al. (1999).
        category_alpha : float, default 0.05
            Significance level for both category tests — the ETS scheme's
            own literature-standard .05, independent of `alpha`.
        no_of_samples : int, default 500
            Bootstrap resamples for each group's std_errors() call.
        seed : int or None, default None
            Seed passed through to each group's internal std_errors() call.
            None draws fresh entropy each call.
        plot : bool, default False
            If True, calls plot_anchor_selection() for each focal group
            that has a selection table (selection_method in ('wald',
            'robust_z')) and stores the resulting figures in dif_plots.
            Defaults to False (unlike calibrate_anchor's plot=True) since
            dif_test can compare multiple focal groups against the
            reference in one
            call, and auto-rendering several plot windows at once isn't
            the right default.
        plot_kwargs : dict or None, default None
            Extra keyword arguments forwarded to plot_anchor_selection()
            for every focal group when plot=True (e.g. filename, xmin/
            xmax, title).
        warm_corr, tolerance, max_iters, ext_score_adjustment : floats
            Person estimation kwargs, passed through to group models. Only
            actually used when test in ('lr', 'both') or omnibus=True.
        constant, method, matrix_power, log_lik_tol : floats
            Calibration kwargs passed to group models.

        Attributes set
        --------------
        dif_table : pandas.DataFrame
            One row per (item, focal group) pair. Columns: 'Group'
            (focal group value), 'Reference' / 'Focal' / 'Focal (purified)'
            (item location estimates), 'Difference' (purified focal -
            reference), 'SE', 'z', 'p', 'p (corrected)', 'Selected' (used
            to define the purified scale), 'Flagged' (Wald p and logit-
            difference thresholds both met). If welch=True, also 'df'
            (Welch-Satterthwaite), and 'p'/'p (corrected)' are t-based
            rather than normal-based ('z' is unchanged either way). If
            test is 'lr' or 'both', also: 'LR', 'p_LR', 'p_LR (corrected)',
            'Flagged_LR'. If category=True, also 'Category' ('A', 'B+',
            'B-', 'C+', or 'C-').
        dif_omnibus_table : pandas.DataFrame or None
            One row per focal group: 'LR', 'df', 'p', 'Flagged' — Andersen-
            style joint test of every common item at once (item-location
            component only). None if omnibus=False.
        threshold_dif_table : pandas.DataFrame
            Differential category functioning (DCF), tested via category
            *widths* (thresholds[k+1] - thresholds[k]) rather than raw
            threshold locations — see component 2 above. One row per
            (category, focal group) pair, where Category k is the width
            between threshold k and k+1 (1..max_score-1, one fewer than
            the number of thresholds). Columns: 'Group', 'Category',
            'Reference' / 'Focal' (category widths), 'Difference', 'SE'
            (from cat_width_se), 'z', 'p', 'p (corrected)', 'Flagged'.
            Corrected independently of dif_table. If welch=True, also 'df'
            (Welch-Satterthwaite), and 'p'/'p (corrected)' are t-based
            rather than normal-based.
        dif_reference : the reference group value.
        dif_covariate : the covariate column name used.
        dif_reference_model : RSM
            Fitted model for the reference group.
        dif_focal_models : dict {focal_value: RSM}
            Fitted models for each focal group.
        dif_tc : dict {focal_value: float}
            Item-scale translation constant applied to each focal group.
        dif_anchor_selection : dict {focal_value: pandas.DataFrame or None}
            Per-item selection diagnostics for each focal group (None if
            selection_method='none').
        dif_plots : dict {focal_value: matplotlib.figure.Figure or None}
            plot_anchor_selection() figure for each focal group, if
            plot=True and a robust-selection table exists for that group;
            None otherwise.
        dif_group_sizes : dict {group_value: int}
            Non-extreme, non-missing-covariate N used for each group.
        """
        from raschpy.rsm import RSM

        if correction not in ("bh", "bonferroni", None):
            raise ValueError("correction must be 'bh', 'bonferroni', or None")
        if selection_method not in ("wald", "robust_z", "none"):
            raise ValueError("selection_method must be 'wald', 'robust_z', or 'none'")
        if test not in ("wald", "lr", "both"):
            raise ValueError("test must be 'wald', 'lr', or 'both'")
        if len(category_thresholds) != 2 or category_thresholds[0] >= category_thresholds[1]:
            raise ValueError(
                "category_thresholds must be (b_threshold, c_threshold) with "
                "b_threshold < c_threshold."
            )
        if reference_n <= 1:
            raise ValueError("reference_n must be greater than 1.")

        if getattr(self, "exogenous", None) is None:
            raise ValueError(
                "No exogenous data available. Pass exogenous= to the "
                "constructor before calling dif_test()."
            )
        if covariate not in self.exogenous.columns:
            raise ValueError(f"'{covariate}' is not a column in self.exogenous.")

        # Non-extreme persons, matching andersen_lr_test's convention
        scores = self.responses.sum(axis=1)
        max_scores = self.responses.notna().sum(axis=1) * self.max_score
        non_extreme = self.responses.index[(scores > 0) & (scores < max_scores)]

        cov_values = self.exogenous.loc[non_extreme, covariate].dropna()
        n_missing = len(non_extreme) - len(cov_values)

        levels = cov_values.value_counts()
        if len(levels) < 2:
            raise ValueError(
                f"'{covariate}' has fewer than 2 distinct non-null values "
                f"among non-extreme persons — cannot run a DIF test."
            )

        if reference is None:
            reference = levels.index[0]
        elif reference not in levels.index:
            raise ValueError(f"reference='{reference}' is not a value of '{covariate}'.")

        focal_levels = [lvl for lvl in levels.index if lvl != reference]
        if len(focal_levels) > 1:
            warnings.warn(
                f"'{covariate}' has {len(levels)} levels. dif_test() compares "
                f"each focal level to the reference level '{reference}' "
                f"individually (reference-group design); all-pairwise "
                f"comparisons between non-reference levels are not supported.",
                UserWarning,
                stacklevel=2,
            )

        group_sizes = {reference: int(levels.loc[reference])}
        for focal in focal_levels:
            group_sizes[focal] = int(levels.loc[focal])

        if n_missing > 0:
            warnings.warn(
                f"{n_missing} non-extreme person(s) have a missing value for "
                f"'{covariate}' and were excluded. Group sizes used: "
                f"{group_sizes}.",
                UserWarning,
                stacklevel=2,
            )

        needs_ll = test in ("lr", "both") or omnibus
        pe_kw = dict(
            warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
            ext_score_adjustment=ext_score_adjustment,
        )

        ref_idx = cov_values.index[cov_values == reference]
        ref_model = RSM(self.responses.loc[ref_idx], max_score=self.max_score)
        ref_model.calibrate(
            constant=constant, threshold_constant=threshold_constant, method=method, matrix_power=matrix_power,
            log_lik_tol=log_lik_tol,
        )
        ref_model.std_errors(
            no_of_samples=no_of_samples, constant=constant, threshold_constant=threshold_constant, method=method,
            matrix_power=matrix_power, log_lik_tol=log_lik_tol, seed=seed,
        )
        if needs_ll:
            ref_model.person_estimates(**pe_kw)
            ll_ref = ref_model._log_likelihood()

        all_item_rows = []
        all_threshold_rows = []
        focal_models = {}
        tc_dict = {}
        anchor_selection_dict = {}
        plot_dict = {}
        omnibus_rows = {}

        for focal in focal_levels:
            focal_idx = cov_values.index[cov_values == focal]
            focal_model = RSM(self.responses.loc[focal_idx], max_score=self.max_score)
            focal_model.calibrate(
                constant=constant, threshold_constant=threshold_constant, method=method, matrix_power=matrix_power,
                log_lik_tol=log_lik_tol,
            )
            focal_model.std_errors(
                no_of_samples=no_of_samples, constant=constant, threshold_constant=threshold_constant, method=method,
                matrix_power=matrix_power, log_lik_tol=log_lik_tol, seed=seed,
            )
            if needs_ll:
                focal_model.person_estimates(**pe_kw)
                ll_focal = focal_model._log_likelihood()

            # --- Item-location DIF ---
            if selection_method == "wald":
                anchor_result = self._wald_anchor_selection(
                    ref_model.items, focal_model.items, focal_model.item_se,
                    alpha=wald_alpha, min_anchors=min_anchors,
                )
                tc = anchor_result["tc"]
                selected = anchor_result["selected_anchors"]
                anchor_selection_dict[focal] = anchor_result["table"]
            elif selection_method == "robust_z":
                anchor_result = self._robust_anchor_selection(
                    ref_model.items, focal_model.items,
                    corr_tol=corr_tol, sd_ratio_tol=sd_ratio_tol,
                    min_anchors=min_anchors,
                )
                tc = anchor_result["tc"]
                selected = anchor_result["selected_anchors"]
                anchor_selection_dict[focal] = anchor_result["table"]
            else:
                common = ref_model.items.index.intersection(focal_model.items.index)
                tc = (
                    ref_model.items.loc[common].mean()
                    - focal_model.items.loc[common].mean()
                )
                selected = common
                anchor_selection_dict[focal] = None

            focal_shifted = focal_model.items + tc

            common_items = ref_model.items.index.intersection(focal_model.items.index)
            diff = focal_shifted.loc[common_items] - ref_model.items.loc[common_items]

            ref_se_item = ref_model.item_se.loc[common_items]
            focal_se_item = focal_model.item_se.loc[common_items]
            ref_n_item = ref_model.no_of_persons
            focal_n_item = focal_model.no_of_persons
            if size_adjust:
                ref_se_item = ref_se_item * np.sqrt(ref_n_item / reference_n)
                focal_se_item = focal_se_item * np.sqrt(focal_n_item / reference_n)
                ref_n_item = focal_n_item = reference_n

            se = np.sqrt(ref_se_item ** 2 + focal_se_item ** 2)
            if welch:
                z_vals, df_vals, p_vals = self._welch_satterthwaite(
                    diff.values,
                    ref_se_item.values, ref_n_item,
                    focal_se_item.values, focal_n_item,
                )
                z = pd.Series(z_vals, index=common_items)
                item_welch_df = pd.Series(df_vals, index=common_items)
                p = pd.Series(p_vals, index=common_items)
            else:
                z = diff / se
                p = pd.Series(2 * norm.sf(np.abs(z.values)), index=common_items)

            if correction == "bonferroni":
                p_corrected = (p * len(p)).clip(upper=1.0)
            elif correction == "bh":
                p_corrected = self._bh_correction(p)
            else:
                p_corrected = p

            flagged = (p_corrected < alpha) & (diff.abs() >= logit_threshold)

            if category:
                b_thr, c_thr = category_thresholds
                abs_diff = diff.abs()
                boundary_stat = (abs_diff.values - b_thr) / se.values
                if welch:
                    p_boundary = pd.Series(
                        t_dist.cdf(-boundary_stat, item_welch_df.values), index=common_items
                    )
                else:
                    p_boundary = pd.Series(norm.cdf(-boundary_stat), index=common_items)

                item_category_col = []
                for item in common_items:
                    if abs_diff[item] >= c_thr and p_boundary[item] < category_alpha:
                        base = "C"
                    elif abs_diff[item] >= b_thr and p[item] < category_alpha:
                        base = "B"
                    else:
                        base = "A"
                    item_category_col.append(
                        base if base == "A" else base + ("+" if diff[item] > 0 else "-")
                    )
                item_category_col = pd.Series(item_category_col, index=common_items)

            if omnibus:
                combined_idx = ref_idx.append(focal_idx)
                m_full = RSM(self.responses.loc[combined_idx], max_score=self.max_score)
                m_full.calibrate(
                    constant=constant, threshold_constant=threshold_constant, method=method, matrix_power=matrix_power,
                    log_lik_tol=log_lik_tol,
                )
                m_full.person_estimates(**pe_kw)
                ll_full = m_full._log_likelihood()
                pooled_ref_persons = m_full.persons.reindex(ref_model.responses.index)
                pooled_focal_persons = m_full.persons.reindex(focal_model.responses.index)
                ll_ref_omni = ref_model._log_likelihood(persons=pooled_ref_persons)
                ll_focal_omni = focal_model._log_likelihood(persons=pooled_focal_persons)
                lr_omni = max(0.0, -2 * (ll_full - (ll_ref_omni + ll_focal_omni)))
                df_omni = len(common_items) - 1
                p_omni = float(chi2.sf(lr_omni, df_omni))
                omnibus_rows[focal] = {
                    "LR": lr_omni, "df": df_omni, "p": p_omni,
                    "Flagged": p_omni < alpha,
                }

            if test in ("lr", "both"):
                ll_h1 = ll_ref + ll_focal
                ref_scratch = RSM(ref_model.responses, max_score=self.max_score)
                ref_scratch.thresholds = ref_model.thresholds
                focal_scratch = RSM(focal_model.responses, max_score=self.max_score)
                focal_scratch.thresholds = focal_model.thresholds
                lr_rows = {}
                for item in common_items:
                    w_ref = 1.0 / ref_model.item_se[item] ** 2
                    w_focal = 1.0 / focal_model.item_se[item] ** 2
                    pooled = (
                        w_ref * ref_model.items[item] + w_focal * focal_shifted[item]
                    ) / (w_ref + w_focal)
                    pooled_focal_scale = pooled - tc

                    ref_scratch.items = ref_model.items.copy()
                    ref_scratch.items[item] = pooled
                    ref_scratch.person_estimates(**pe_kw)
                    ll_ref_h0 = ref_scratch._log_likelihood()

                    focal_scratch.items = focal_model.items.copy()
                    focal_scratch.items[item] = pooled_focal_scale
                    focal_scratch.person_estimates(**pe_kw)
                    ll_focal_h0 = focal_scratch._log_likelihood()

                    lr_i = max(0.0, 2 * (ll_h1 - (ll_ref_h0 + ll_focal_h0)))
                    lr_rows[item] = {"LR": lr_i, "p_LR": float(chi2.sf(lr_i, 1))}

                lr_table = pd.DataFrame(lr_rows).T.loc[common_items]
                if correction == "bonferroni":
                    lr_table["p_LR (corrected)"] = (
                        lr_table["p_LR"] * len(lr_table)
                    ).clip(upper=1.0)
                elif correction == "bh":
                    lr_table["p_LR (corrected)"] = self._bh_correction(lr_table["p_LR"])
                else:
                    lr_table["p_LR (corrected)"] = lr_table["p_LR"]
                lr_table["Flagged_LR"] = (lr_table["p_LR (corrected)"] < alpha) & (
                    diff.abs() >= logit_threshold
                )

            item_table = pd.DataFrame(
                {
                    "Group": focal,
                    "Reference": ref_model.items.loc[common_items],
                    "Focal": focal_model.items.loc[common_items],
                    "Focal (purified)": focal_shifted.loc[common_items],
                    "Difference": diff,
                    "SE": se,
                    "z": z,
                    "p": p,
                    "p (corrected)": p_corrected,
                    "Selected": common_items.isin(selected),
                    "Flagged": flagged,
                },
                index=common_items,
            )
            item_table.index.name = "Item"
            if welch:
                item_table["df"] = item_welch_df
            if category:
                item_table["Category"] = item_category_col
            if test in ("lr", "both"):
                item_table = item_table.join(lr_table)

            # --- Threshold-structure DIF (DCF: category widths, shared vector,
            # no purification) ---
            # Category k's width (thresholds[k+1] - thresholds[k]) is the natural
            # unit for category-structure DIF, not raw threshold locations — locations
            # are partial sums of widths, so a single genuine width change cascades
            # into every downstream threshold location, smearing/mis-localising the
            # signal if tested directly. cat_width_se (from std_errors()) differences
            # *within* each bootstrap resample before taking the std, so it already
            # reflects Cov(threshold_k, threshold_{k+1}) with no extra covariance
            # term needed.
            ref_widths = ref_model.thresholds.diff().dropna()
            focal_widths = focal_model.thresholds.diff().dropna()
            ref_widths.index = focal_widths.index = range(1, len(ref_widths) + 1)
            thr_index = ref_widths.index
            thr_diff = focal_widths - ref_widths

            ref_se_thr = ref_model.cat_width_se
            focal_se_thr = focal_model.cat_width_se
            ref_n_thr = ref_model.no_of_persons
            focal_n_thr = focal_model.no_of_persons
            if size_adjust:
                ref_se_thr = ref_se_thr * np.sqrt(ref_n_thr / reference_n)
                focal_se_thr = focal_se_thr * np.sqrt(focal_n_thr / reference_n)
                ref_n_thr = focal_n_thr = reference_n

            thr_se = pd.Series(
                np.sqrt(ref_se_thr ** 2 + focal_se_thr ** 2).values, index=thr_index
            )
            if welch:
                thr_z_vals, thr_df_vals, thr_p_vals = self._welch_satterthwaite(
                    thr_diff.values, ref_se_thr.values, ref_n_thr,
                    focal_se_thr.values, focal_n_thr,
                )
                thr_z = pd.Series(thr_z_vals, index=thr_index)
                thr_welch_df = pd.Series(thr_df_vals, index=thr_index)
                thr_p = pd.Series(thr_p_vals, index=thr_index)
            else:
                thr_z = thr_diff / thr_se
                thr_p = pd.Series(2 * norm.sf(np.abs(thr_z.values)), index=thr_index)

            if correction == "bonferroni":
                thr_p_corrected = (thr_p * len(thr_p)).clip(upper=1.0)
            elif correction == "bh":
                thr_p_corrected = self._bh_correction(thr_p)
            else:
                thr_p_corrected = thr_p

            thr_flagged = (thr_p_corrected < alpha) & (thr_diff.abs() >= logit_threshold)

            threshold_table = pd.DataFrame(
                {
                    "Group": focal,
                    "Category": thr_index,
                    "Reference": ref_widths,
                    "Focal": focal_widths,
                    "Difference": thr_diff,
                    "SE": thr_se,
                    "z": thr_z,
                    "p": thr_p,
                    "p (corrected)": thr_p_corrected,
                    "Flagged": thr_flagged,
                },
                index=thr_index,
            )
            if welch:
                threshold_table["df"] = thr_welch_df

            all_item_rows.append(item_table)
            all_threshold_rows.append(threshold_table)
            focal_models[focal] = focal_model
            tc_dict[focal] = tc

            if plot and anchor_selection_dict[focal] is not None:
                plot_dict[focal] = self.plot_anchor_selection(
                    anchor_selection_dict[focal], **(plot_kwargs or {})
                )
            else:
                plot_dict[focal] = None

        self.dif_table = pd.concat(all_item_rows)
        self.dif_omnibus_table = (
            pd.DataFrame(omnibus_rows).T if omnibus else None
        )
        self.threshold_dif_table = pd.concat(all_threshold_rows, ignore_index=True)
        self.dif_reference = reference
        self.dif_covariate = covariate
        self.dif_reference_model = ref_model
        self.dif_focal_models = focal_models
        self.dif_tc = tc_dict
        self.dif_anchor_selection = anchor_selection_dict
        self.dif_plots = plot_dict
        self.dif_group_sizes = group_sizes

    def model_selection(
        self,
        test="AIC",
        aic_sig_test=True,
        alpha=0.05,
        min_effect=0,
        sampling="dynamic",
        warm_corr=True,
        tolerance=0.00001,
        max_iters=100,
        ext_score_adjustment=0.5,
        constant=None,
        threshold_constant=None,
        method="log-lik",
        log_lik_tol=0.000001,
        seed=None,
    ):
        """
        Compare RSM against PCM using a likelihood ratio test, AIC, or BIC.

        RSM is the constrained (nested) model; PCM is unconstrained.

        Parameters
        ----------
        test : str, default 'LR'
            Test to run: 'LR', 'AIC', or 'BIC'.
        aic_sig_test : bool, default False
            When True (requires test='AIC'), applies a significance test with
            RSM as the null hypothesis. The p-value is the relative likelihood
            of RSM vs PCM: p = e^(-Δ/2) where Δ = AIC_RSM - AIC_PCM. PCM is
            preferred only if p < alpha; otherwise RSM is retained as default.
        alpha : float, default 0.05
            Significance level used to decide the preferred model for the
            LR test (RSM is the null; PCM is preferred if p < alpha) and,
            when aic_sig_test=True, for the AIC relative-likelihood test.
            Not used by BIC, which has no formal significance test and
            simply prefers whichever model has the lower BIC.
        min_effect : float, default 0
            Minimum effect size (logits) required to prefer PCM over RSM,
            in addition to the significance/IC test for each test type.
            The effect is max|PCM.thresholds - RSM.thresholds| -- the
            largest absolute deviation of any single item's own
            (item-difficulty-centred) threshold shape from RSM's single
            shared threshold vector. Applied as PCM preferred only if the
            test's own condition holds AND effect >= min_effect; RSM is
            retained otherwise. Default 0 disables (never gates).
        sampling : None, 'dynamic', or int, default 'dynamic'
            Controls subsampling of non-extreme persons before computing
            log-likelihoods (parameters are always estimated on the full
            data; only the LL evaluation is subsampled). Not primarily
            about speed: at very large N, LR/AIC/BIC all become biased
            toward the more complex (PCM) model regardless of whether the
            difference is practically meaningful, since log-likelihood
            differences scale with N. Capping the effective N keeps the
            comparison from being dominated by sample size. None disables
            subsampling. 'dynamic' uses T = min(20*(I-1)*(m-1), 1500); an
            integer fixes T directly. When n <= T, sampling is skipped.
            Applies to all three tests.
        warm_corr, tolerance, max_iters, ext_score_adjustment : floats
            Person estimation kwargs.
        constant, method, log_lik_tol : floats
            Calibration kwargs.
        seed : int or None, default None
            Seed for the person subsampling RNG (only used when sampling
            triggers). Pass an int for reproducible LL evaluation; None
            (default) draws fresh entropy.

        Attributes set
        --------------
        model_comparison_rsm_pcm_effect : max|PCM.thresholds - RSM.thresholds| (see min_effect).
        model_comparison_rsm_pcm_lr, _df, _p, _lr_preferred, _lr_summary : LR test results.
        model_comparison_rsm_pcm_aic, _aic_preferred, _aic_summary : AIC results.
        model_comparison_rsm_pcm_aic_p : relative likelihood p-value (aic_sig_test only).
        model_comparison_rsm_pcm_bic, _bic_preferred, _bic_summary : BIC results.
        """
        from raschpy.pcm import PCM

        if test not in ("LR", "AIC", "BIC"):
            raise ValueError("test must be 'LR', 'AIC', or 'BIC'")
        if sampling is not None and sampling != "dynamic" and not isinstance(sampling, int):
            raise ValueError("sampling must be None, 'dynamic', or an integer")

        if not hasattr(self, "thresholds"):
            self.calibrate(constant=constant, threshold_constant=threshold_constant, method=method, log_lik_tol=log_lik_tol)
        if not hasattr(self, "persons"):
            self.person_estimates(
                warm_corr=warm_corr,
                tolerance=tolerance,
                max_iters=max_iters,
                ext_score_adjustment=ext_score_adjustment,
            )

        scores = self.responses.sum(axis=1)
        max_scores = self.responses.notna().sum(axis=1) * self.max_score
        non_extreme_mask = (scores > 0) & (scores < max_scores)
        n_persons = int(non_extreme_mask.sum())

        # Fit PCM on full data (parameters used for both full and sampled LL)
        pcm = PCM(self.responses)
        # PCM takes a single scalar constant; use the item-stage value.
        pcm.calibrate(constant=(0.1 if constant is None else constant), method=method, log_lik_tol=log_lik_tol)
        pcm.person_estimates(
            warm_corr=warm_corr,
            tolerance=tolerance,
            max_iters=max_iters,
            ext_score_adjustment=ext_score_adjustment,
        )

        # Determine responses for LL computation (sample from non-extreme persons)
        ll_responses = None  # None → _log_likelihood uses self.responses
        n_ll = n_persons
        if sampling is not None:
            T = (
                min(20 * (self.no_of_items - 1) * (self.max_score - 1), 1500)
                if sampling == "dynamic"
                else int(sampling)
            )
            if n_persons > T:
                rng = np.random.default_rng(seed)
                non_extreme_idx = self.responses.index[non_extreme_mask]
                sampled_idx = rng.choice(non_extreme_idx, size=T, replace=False)
                ll_responses = self.responses.loc[sampled_idx]
                n_ll = T

        ll_rsm = self._log_likelihood(responses=ll_responses)
        ll_pcm = pcm._log_likelihood(responses=ll_responses)

        k_rsm = (self.no_of_items - 1) + (self.max_score - 1)
        k_pcm = int(pcm.thresholds_uncentred.notna().sum().sum()) - 1

        effect = float(pcm.thresholds.sub(self.thresholds, axis=1).abs().to_numpy().max())
        self.model_comparison_rsm_pcm_effect = effect

        if test == "LR":
            lr = -2 * (ll_rsm - ll_pcm)
            if lr < 0:
                warnings.warn(
                    "RSM vs PCM LR statistic is negative due to PAIR estimation "
                    "approximation and has been floored at 0. This indicates no "
                    "evidence that PCM fits better than RSM.",
                    UserWarning,
                )
                lr = 0.0
            df = (self.no_of_items - 1) * (self.max_score - 1)
            p = float(chi2.sf(lr, df))
            preferred = "PCM" if (p < alpha and effect >= min_effect) else "RSM"
            self.model_comparison_rsm_pcm_lr = lr
            self.model_comparison_rsm_pcm_df = df
            self.model_comparison_rsm_pcm_p = p
            self.model_comparison_rsm_pcm_lr_preferred = preferred
            self.model_comparison_rsm_pcm_lr_summary = pd.Series(
                {"LR statistic": lr, "df": df, "p-value": p, "Effect": effect, "Preferred": preferred},
                name="RSM vs PCM LR test",
            )

        elif test == "AIC":
            aic_pcm = 2 * k_pcm - 2 * ll_pcm
            aic_rsm = 2 * k_rsm - 2 * ll_rsm
            self.model_comparison_rsm_pcm_aic = {"PCM": aic_pcm, "RSM": aic_rsm}

            if aic_sig_test:
                delta = aic_rsm - aic_pcm
                aic_p = float(np.exp(-abs(delta) / 2))
                preferred = "PCM" if (delta > 0 and aic_p < alpha and effect >= min_effect) else "RSM"
                self.model_comparison_rsm_pcm_aic_p = aic_p
                self.model_comparison_rsm_pcm_aic_preferred = preferred
                self.model_comparison_rsm_pcm_aic_summary = pd.Series(
                    {
                        "PCM AIC": aic_pcm,
                        "RSM AIC": aic_rsm,
                        "p-value": aic_p,
                        "Effect": effect,
                        "Preferred": preferred,
                    },
                    name="RSM vs PCM AIC comparison",
                )
            else:
                preferred = "PCM" if (aic_pcm < aic_rsm and effect >= min_effect) else "RSM"
                self.model_comparison_rsm_pcm_aic_preferred = preferred
                self.model_comparison_rsm_pcm_aic_summary = pd.Series(
                    {"PCM AIC": aic_pcm, "RSM AIC": aic_rsm, "Effect": effect, "Preferred": preferred},
                    name="RSM vs PCM AIC comparison",
                )

        elif test == "BIC":
            bic_pcm = k_pcm * np.log(n_ll) - 2 * ll_pcm
            bic_rsm = k_rsm * np.log(n_ll) - 2 * ll_rsm
            preferred = "PCM" if (bic_pcm < bic_rsm and effect >= min_effect) else "RSM"
            self.model_comparison_rsm_pcm_bic = {"PCM": bic_pcm, "RSM": bic_rsm}
            self.model_comparison_rsm_pcm_bic_preferred = preferred
            self.model_comparison_rsm_pcm_bic_summary = pd.Series(
                {"PCM BIC": bic_pcm, "RSM BIC": bic_rsm, "Effect": effect, "Preferred": preferred},
                name="RSM vs PCM BIC comparison",
            )

    def res_corr_analysis(
        self,
        warm_corr=True,
        tolerance=0.00001,
        max_iters=100,
        ext_score_adjustment=0.5,
        constant=None,
        threshold_constant=None,
        method="log-lik",
        matrix_power=None,
        log_lik_tol=0.000001,
        no_of_samples=500,
        interval=None,
        se=True,
    ):
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
        constant : float or None, default None
            Smoothing constant, passed to calibrate(); None resolves to 0.1.
        threshold_constant : float or None, default None
            Passed to calibrate() as the Stage 2 CPAT constant; None
            follows `constant` (0.03 when constant is None). See calibrate().
        method : str, default 'log-lik'
            Priority vector extraction method.
        matrix_power : int or None, default None
            Matrix power for calibration. None resolves to 0 for
            method='log-lik' (no powering), else 5.
        log_lik_tol : float, default 0.000001
            Convergence tolerance for calibration.
        no_of_samples : int, default 500
            Bootstrap samples. Unused if se=False.
        interval : float or None, default None
            CI width. Unused if se=False.
        se : bool, default True
            Passed through to the internal fit_statistics() call (only
            used if not already computed). If False, skips the bootstrap
            entirely — this analysis's own output (residual correlations,
            PCA) does not depend on it, so se=False is purely a speed-up
            (e.g. for repeated simulation runs) with no effect on the
            output.

        Attributes set
        --------------
        residual_correlations : pandas.DataFrame
            Item-by-item correlation matrix of standardised residuals.
        eigenvectors, eigenvalues, variance_explained, loadings : DataFrame or None
            PCA results. None if PCA fails.
        pca_fail : bool
            True only if PCA raises an exception.
        """
        if not hasattr(self, "std_residual_df"):
            self.fit_statistics(
                se=se,
                warm_corr=warm_corr,
                tolerance=tolerance,
                max_iters=max_iters,
                ext_score_adjustment=ext_score_adjustment,
                constant=constant,
                threshold_constant=threshold_constant,
                method=method,
                matrix_power=matrix_power,
                log_lik_tol=log_lik_tol,
                no_of_samples=no_of_samples,
                interval=interval,
            )

        self.residual_correlations = self.residual_df.corr(numeric_only=False)
        pca = PCA()
        try:
            pca.fit(self.std_residual_df.corr())
            n = (
                self.no_of_items - 1
            )  # rank of correlation matrix is n-1; drop zero eigenvalue
            pc_labels = [f"PC {pc + 1}" for pc in range(n)]
            self.eigenvectors = pd.DataFrame(
                pca.components_[:n, :],
                index=pc_labels,
                columns=[f"Eigenvector {pc + 1}" for pc in range(self.no_of_items)],
            )
            self.eigenvalues = pd.DataFrame(
                pca.explained_variance_[:n], index=pc_labels, columns=["Eigenvalue"]
            )
            self.variance_explained = pd.DataFrame(
                pca.explained_variance_ratio_[:n],
                index=pc_labels,
                columns=["Variance explained"],
            )
            self.loadings = pd.DataFrame(
                self.eigenvectors.values.T * (pca.explained_variance_[:n] ** 0.5),
                index=self.responses.columns,
                columns=pc_labels,
            )
        except Exception:
            self.pca_fail = True
            warnings.warn(
                "PCA of standardised residuals failed. "
                "Eigenvectors and loadings set to None.",
                UserWarning,
                stacklevel=2,
            )
            self.eigenvectors = self.eigenvalues = None
            self.variance_explained = self.loadings = None

    # ------------------------------------------------------------------
    # Output tables
    # ------------------------------------------------------------------

    def item_stats_df(
        self,
        full=False,
        zstd=False,
        point_measure_corr=False,
        dp=3,
        se=True,
        warm_corr=True,
        tolerance=0.00001,
        max_iters=100,
        ext_score_adjustment=0.5,
        method="log-lik",
        constant=None,
        threshold_constant=None,
        matrix_power=None,
        log_lik_tol=0.000001,
        no_of_samples=500,
        interval=None,
        seed=None,
    ):
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
        se : bool, default True
            If True, computes and includes the SE column (and CI bound
            columns, if interval is set). If False, skips the bootstrap
            entirely — useful when only Infit/Outfit MS are needed (e.g.
            repeated simulation runs), since those do not depend on the
            bootstrap. Forces interval to None when False.
        warm_corr : bool, default True
            Warm bias correction.
        tolerance : float, default 0.00001
            Newton-Raphson convergence tolerance.
        max_iters : int, default 100
            Maximum iterations.
        ext_score_adjustment : float, default 0.5
            Extreme score adjustment.
        method : str, default 'log-lik'
            Priority vector extraction method.
        constant : float or None, default None
            Smoothing constant, passed to calibrate(); None resolves to 0.1.
        threshold_constant : float or None, default None
            Passed to calibrate() as the Stage 2 CPAT constant; None
            follows `constant` (0.03 when constant is None). See calibrate().
        matrix_power : int or None, default None
            Matrix power for calibration. None resolves to 0 for
            method='log-lik' (no powering), else 5.
        log_lik_tol : float, default 0.000001
            Log-likelihood tolerance for calibration.
        no_of_samples : int, default 500
            Bootstrap samples. Unused if se=False.
        interval : float or None, default None
            CI width; if provided, percentile bound columns included.
            Ignored if se=False.
        seed : int or None, default None
            Seed passed through to the internal std_errors()/fit_statistics()
            calls (only used if not already computed). None draws fresh
            entropy each call.

        Attributes set
        --------------
        item_stats : pandas.DataFrame
            Item statistics with items as rows. Always contains Estimate,
            Count, Facility, Infit MS, Outfit MS. Optional: SE and CI
            bounds (if se=True).
        """

        if full:
            zstd = True
            point_measure_corr = True
            if interval is None:
                interval = 0.95

        if not se:
            interval = None

        if se and (
            not hasattr(self, "threshold_se")
            or (self.threshold_low is None and interval is not None)
        ):
            self.std_errors(
                interval=interval,
                no_of_samples=no_of_samples,
                constant=constant,
                threshold_constant=threshold_constant,
                method=method,
                matrix_power=matrix_power,
                log_lik_tol=log_lik_tol,
                seed=seed,
            )
        if not hasattr(self, "item_infit_ms"):
            self.fit_statistics(
                se=se,
                warm_corr=warm_corr,
                tolerance=tolerance,
                max_iters=max_iters,
                ext_score_adjustment=ext_score_adjustment,
                method=method,
                constant=constant,
                threshold_constant=threshold_constant,
                matrix_power=matrix_power,
                log_lik_tol=log_lik_tol,
                no_of_samples=no_of_samples,
                interval=interval,
                seed=seed,
            )

        stats = pd.DataFrame(index=self.responses.columns)
        stats["Estimate"] = self.items.round(dp)
        if se:
            stats["SE"] = self.item_se.round(dp)
            if interval is not None:
                stats[f"{round((1 - interval) * 50, 1)}%"] = self.item_low.round(dp)
                stats[f"{round((1 + interval) * 50, 1)}%"] = self.item_high.round(dp)
        stats["Count"] = self.response_counts.astype(int)
        stats["Facility"] = self.item_facilities.round(dp)
        stats["Infit MS"] = self.item_infit_ms.round(dp)
        if zstd:
            stats["Infit Z"] = self.item_infit_zstd.round(dp)
        stats["Outfit MS"] = self.item_outfit_ms.round(dp)
        if zstd:
            stats["Outfit Z"] = self.item_outfit_zstd.round(dp)
        if point_measure_corr:
            stats["PM corr"] = self.point_measure.round(dp)
            stats["Exp PM corr"] = self.exp_point_measure.round(dp)
        self.item_stats = stats

    def threshold_stats_df(
        self,
        full=False,
        zstd=False,
        disc=False,
        point_measure_corr=False,
        dp=3,
        warm_corr=True,
        tolerance=0.00001,
        max_iters=100,
        ext_score_adjustment=0.5,
        method="log-lik",
        constant=None,
        threshold_constant=None,
        no_of_samples=500,
        interval=None,
    ):
        """
        Build and store the threshold statistics summary table.

        Auto-triggers fit_statistics() if not yet run. Reports statistics for
        the max_score shared Rasch-Andrich thresholds (thresholds[1..max_score]).
        Unlike PCM, RSM has one shared
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
        method : str, default 'log-lik'
            Priority vector extraction method.
        constant : float or None, default None
            Smoothing constant, passed to calibrate(); None resolves to 0.1.
        threshold_constant : float or None, default None
            Passed to calibrate() as the Stage 2 CPAT constant; None
            follows `constant` (0.03 when constant is None). See calibrate().
        no_of_samples : int, default 500
            Bootstrap samples.
        interval : float or None, default None
            CI width.

        Attributes set
        --------------
        threshold_stats : pandas.DataFrame
            Threshold statistics, rows Threshold 1..Threshold max_score.
            Always contains Estimate, SE, Infit MS, Outfit MS. See also
            category_stats_df() for category-*width* statistics — the
            physically meaningful, full-rank quantity for step-structure
            questions (a zero-summed threshold vector only has
            max_score-1 true degrees of freedom, so per-threshold SEs
            here are correlated, not independent).
        """

        if full:
            zstd = True
            disc = True
            point_measure_corr = True
            if interval is None:
                interval = 0.95

        if not hasattr(self, "threshold_infit_ms"):
            self.fit_statistics(
                warm_corr=warm_corr,
                tolerance=tolerance,
                max_iters=max_iters,
                ext_score_adjustment=ext_score_adjustment,
                method=method,
                constant=constant,
                threshold_constant=threshold_constant,
                no_of_samples=no_of_samples,
                interval=interval,
            )

        idx = [f"Threshold {t + 1}" for t in range(self.max_score)]
        stats = pd.DataFrame(index=idx)
        stats["Estimate"] = self.thresholds.values.round(dp)
        stats["SE"] = self.threshold_se.round(dp)
        if interval is not None:
            stats[f"{round((1 - interval) * 50, 1)}%"] = self.threshold_low.round(dp)
            stats[f"{round((1 + interval) * 50, 1)}%"] = self.threshold_high.round(dp)
        stats["Infit MS"] = self.threshold_infit_ms.values.round(dp)
        if zstd:
            stats["Infit Z"] = self.threshold_infit_zstd.values.round(dp)
        stats["Outfit MS"] = self.threshold_outfit_ms.values.round(dp)
        if zstd:
            stats["Outfit Z"] = self.threshold_outfit_zstd.values.round(dp)
        if disc:
            stats["Discrim"] = self.threshold_discrimination.values.round(dp)
        if point_measure_corr:
            stats["PM corr"] = self.threshold_point_measure.values.round(dp)
            stats["Exp PM corr"] = self.threshold_exp_point_measure.values.round(dp)
        self.threshold_stats = stats

    def category_stats_df(
        self,
        dp=3,
        constant=None,
        threshold_constant=None,
        method="log-lik",
        matrix_power=None,
        log_lik_tol=0.000001,
        no_of_samples=500,
        interval=None,
        seed=None,
    ):
        """
        Build and store the category width statistics summary table.

        Reports category *widths* (thresholds[k+1] - thresholds[k]) rather
        than raw threshold locations — see threshold_stats_df for the full
        rationale: a zero-summed threshold vector has only max_score-1 true
        degrees of freedom, so the max_score per-threshold SEs threshold_
        stats_df reports are correlated (perfectly, at max_score=2), not
        independent, and understate the true uncertainty of the physically
        meaningful step-structure quantity. Reported *alongside*
        threshold_stats_df's output, not instead of it — threshold-level
        SEs remain the expected, standard report.

        Deliberately lighter than threshold_stats_df: no Infit/Outfit or
        other fit statistics, since those aren't naturally defined for a
        difference of two threshold locations. Auto-triggers calibrate()/
        std_errors() directly if not yet run (not the full, heavier
        fit_statistics()).

        Widths can be negative — a negative width at category k means
        thresholds k and k+1 are disordered (category k is never the most
        likely response at any person location). Prop disordered makes this
        a continuous diagnostic rather than a single point-estimate
        yes/no: the proportion of bootstrap resamples in which that
        category's width was negative — reasonably read as the
        probability that the true category is disordered.

        Parameters
        ----------
        dp : int, default 3
            Decimal places.
        constant, method, matrix_power, log_lik_tol : floats
            Calibration kwargs, used only if calibrate() hasn't already
            been run.
        no_of_samples : int, default 500
            Bootstrap samples, used only if std_errors() hasn't already
            been run.
        interval : float or None, default None
            CI width. If provided, lower/upper percentile columns are
            included — but only if std_errors() is triggered by this call
            or was already run with an interval; it is not retroactively
            added to an existing SE run made without one.
        seed : int or None, default None
            Seed for std_errors(), used only if it hasn't already been
            run.

        Attributes set
        --------------
        category_stats : pandas.DataFrame
            Rows Category 1..Category max_score-1 (one fewer row than
            threshold_stats). Columns: Estimate (the width itself, can be
            negative), SE (from cat_width_se — differenced *within* each
            bootstrap resample before taking the std, so it already
            reflects Cov(threshold_k, threshold_{k+1})), CI bounds if
            interval is not None, Disordered (Estimate < 0), and
            Prop disordered — the bootstrap proportion of resamples with a
            negative width, reasonably read as the probability that the
            true category is disordered. Useful for interpreting
            Disordered in both directions: a True that's only weakly
            supported (Prop disordered close to 0.5) vs. robust, or a
            False that's nonetheless uncertain (Prop disordered not
            small) vs. clear-cut.
        """
        if not hasattr(self, "thresholds"):
            self.calibrate(
                constant=constant, threshold_constant=threshold_constant, method=method, matrix_power=matrix_power,
                log_lik_tol=log_lik_tol,
            )
        if not hasattr(self, "cat_width_se"):
            self.std_errors(
                interval=interval, no_of_samples=no_of_samples,
                constant=constant, threshold_constant=threshold_constant, method=method, matrix_power=matrix_power,
                log_lik_tol=log_lik_tol, seed=seed,
            )

        cat_widths = self.thresholds.diff().dropna()
        cat_widths.index = range(1, self.max_score)
        cat_idx = [f"Category {k}" for k in range(1, self.max_score)]
        stats = pd.DataFrame(index=cat_idx)
        stats["Estimate"] = cat_widths.values.round(dp)
        stats["SE"] = self.cat_width_se.values.round(dp)
        if interval is not None and self.cat_width_low is not None:
            stats[f"{round((1 - interval) * 50, 1)}%"] = np.array(
                list(self.cat_width_low.values())
            ).round(dp)
            stats[f"{round((1 + interval) * 50, 1)}%"] = np.array(
                list(self.cat_width_high.values())
            ).round(dp)
        stats["Disordered"] = cat_widths.values < 0
        stats["Prop disordered"] = (
            (self.cat_width_bootstrap < 0).mean(axis=0).values.round(dp)
        )
        self.category_stats = stats

    def person_stats_df(
        self,
        full=False,
        rsem=False,
        dp=3,
        se=True,
        warm_corr=True,
        tolerance=0.00001,
        max_iters=100,
        ext_score_adjustment=0.5,
        method="log-lik",
        constant=None,
        threshold_constant=None,
    ):
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
        se : bool, default True
            Passed through to the internal fit_statistics() call (only
            used if not already computed). If False, skips the bootstrap
            entirely — this table's own columns (CSEM, RSEM, Infit/Outfit)
            do not depend on it, so se=False is purely a speed-up (e.g.
            for repeated simulation runs) with no effect on the output.
        warm_corr : bool, default True
            Warm bias correction.
        tolerance : float, default 0.00001
            Convergence tolerance.
        max_iters : int, default 100
            Maximum iterations.
        ext_score_adjustment : float, default 0.5
            Extreme score adjustment.
        method : str, default 'log-lik'
            Priority vector extraction method.
        constant : float or None, default None
            Smoothing constant, passed to calibrate(); None resolves to 0.1.
        threshold_constant : float or None, default None
            Passed to calibrate() as the Stage 2 CPAT constant; None
            follows `constant` (0.03 when constant is None). See calibrate().

        Attributes set
        --------------
        person_stats : pandas.DataFrame
            Person statistics with persons as rows. Contains Estimate, CSEM,
            Score, Max score, p, Infit MS, Infit Z, Outfit MS, Outfit Z.
            Optional: RSEM.
        """

        if not hasattr(self, "person_infit_ms"):
            self.fit_statistics(
                se=se,
                warm_corr=warm_corr,
                tolerance=tolerance,
                max_iters=max_iters,
                ext_score_adjustment=ext_score_adjustment,
                method=method,
                constant=constant,
                threshold_constant=threshold_constant,
            )
        if full:
            rsem = True

        idx = self.responses.index
        stats = pd.DataFrame(index=idx)
        stats["Estimate"] = self.persons.round(dp)
        stats["CSEM"] = self.csem_vector.round(dp)
        if rsem:
            stats["RSEM"] = self.rsem_vector.round(dp)
        stats["Score"] = self.responses.sum(axis=1).astype(int)
        stats["Max score"] = (self.responses.count(axis=1) * self.max_score).astype(int)
        stats["p"] = (self.responses.mean(axis=1) / self.max_score).round(dp)

        # BUG FIX: original used .update(dict) which ignores index alignment.
        for col, src in [
            ("Infit MS", self.person_infit_ms),
            ("Infit Z", self.person_infit_zstd),
            ("Outfit MS", self.person_outfit_ms),
            ("Outfit Z", self.person_outfit_zstd),
        ]:
            stats[col] = np.nan
            stats.loc[src.index, col] = src.round(dp).values

        self.person_stats = stats

    def test_stats_df(
        self,
        dp=3,
        warm_corr=True,
        tolerance=0.00001,
        max_iters=100,
        ext_score_adjustment=0.5,
        method="log-lik",
        constant=None,
        threshold_constant=None,
        alpha=False,
        seed=None,
    ):
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
        method : str, default 'log-lik'
            Priority vector extraction method.
        constant : float or None, default None
            Smoothing constant, passed to calibrate(); None resolves to 0.1.
        threshold_constant : float or None, default None
            Passed to calibrate() as the Stage 2 CPAT constant; None
            follows `constant` (0.03 when constant is None). See calibrate().
        alpha : bool, default False
            If True, adds a 'Cronbach alpha' row (Persons column only —
            Items is left NaN, since Cronbach's alpha is a person-side
            reliability statistic). Computed on complete cases; if the
            data contain missing responses, a UserWarning is raised and
            alpha is computed after listwise deletion, which may
            underestimate the true value.
        seed : int or None, default None
            Seed passed through to the internal fit_statistics() call (only
            used if not already computed). None draws fresh entropy.

        Attributes set
        --------------
        test_stats : pandas.DataFrame
            Two-column table (Items, Persons) with rows:
            Mean, SD, Separation ratio, Strata, Reliability.
        """

        if not hasattr(self, "psi"):
            self.fit_statistics(
                warm_corr=warm_corr,
                tolerance=tolerance,
                max_iters=max_iters,
                ext_score_adjustment=ext_score_adjustment,
                method=method,
                constant=constant,
                threshold_constant=threshold_constant,
                seed=seed,
            )

        # RSM test stats have no threshold separation row (thresholds are
        # shared, not item-specific, so threshold ISI is not meaningful here).
        items_col = [self.items.mean(), self.items.std(), self.isi,
                     self.item_strata, self.item_reliability]
        persons_col = [self.persons.mean(), self.persons.std(), self.psi,
                       self.person_strata, self.person_reliability]
        index = ["Mean", "SD", "Separation ratio", "Strata", "Reliability"]

        if alpha:
            items_col.append(np.nan)
            persons_col.append(self._cronbach_alpha())
            index.append("Cronbach alpha")

        self.test_stats = pd.DataFrame(
            {"Items": items_col, "Persons": persons_col}, index=index
        )
        self.test_stats = self.test_stats.round(dp)

    def save_stats(
        self,
        filename,
        format="csv",
        dp=3,
        warm_corr=True,
        tolerance=0.00001,
        max_iters=100,
        ext_score_adjustment=0.5,
        method="log-lik",
        constant=None,
        threshold_constant=None,
        no_of_samples=500,
        interval=None,
    ):
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
        method : str, default 'log-lik'
            Priority vector extraction method.
        constant : float or None, default None
            Smoothing constant, passed to calibrate(); None resolves to 0.1.
        threshold_constant : float or None, default None
            Passed to calibrate() as the Stage 2 CPAT constant; None
            follows `constant` (0.03 when constant is None). See calibrate().
        no_of_samples : int, default 500
            Bootstrap samples.
        interval : float or None, default None
            CI width.
        """

        for attr, method_name, kwargs in [
            (
                "item_stats",
                "item_stats_df",
                dict(
                    dp=dp,
                    warm_corr=warm_corr,
                    tolerance=tolerance,
                    max_iters=max_iters,
                    ext_score_adjustment=ext_score_adjustment,
                    method=method,
                    constant=constant,
                    threshold_constant=threshold_constant,
                    no_of_samples=no_of_samples,
                    interval=interval,
                ),
            ),
            (
                "threshold_stats",
                "threshold_stats_df",
                dict(
                    dp=dp,
                    warm_corr=warm_corr,
                    tolerance=tolerance,
                    max_iters=max_iters,
                    ext_score_adjustment=ext_score_adjustment,
                    method=method,
                    constant=constant,
                    threshold_constant=threshold_constant,
                    no_of_samples=no_of_samples,
                    interval=interval,
                ),
            ),
            (
                "person_stats",
                "person_stats_df",
                dict(
                    dp=dp,
                    warm_corr=warm_corr,
                    tolerance=tolerance,
                    max_iters=max_iters,
                    ext_score_adjustment=ext_score_adjustment,
                    method=method,
                    constant=constant,
                    threshold_constant=threshold_constant,
                ),
            ),
            (
                "test_stats",
                "test_stats_df",
                dict(
                    dp=dp,
                    warm_corr=warm_corr,
                    tolerance=tolerance,
                    max_iters=max_iters,
                    ext_score_adjustment=ext_score_adjustment,
                    method=method,
                    constant=constant,
                    threshold_constant=threshold_constant,
                ),
            ),
        ]:
            if not hasattr(self, attr):
                getattr(self, method_name)(**kwargs)

        if format == "xlsx":
            if not filename.endswith(".xlsx"):
                filename += ".xlsx"
            with pd.ExcelWriter(filename, engine="openpyxl") as writer:
                self.item_stats.to_excel(writer, sheet_name="Item statistics")
                self.threshold_stats.to_excel(writer, sheet_name="Threshold statistics")
                self.person_stats.to_excel(writer, sheet_name="Person statistics")
                self.test_stats.to_excel(writer, sheet_name="Test statistics")
        else:
            if filename.endswith(".csv"):
                filename = filename[:-4]
            self.item_stats.to_csv(f"{filename}_item_stats.csv")
            self.threshold_stats.to_csv(f"{filename}_threshold_stats.csv")
            self.person_stats.to_csv(f"{filename}_person_stats.csv")
            self.test_stats.to_csv(f"{filename}_test_stats.csv")

    def save_residuals(
        self,
        filename,
        format="csv",
        single=True,
        dp=3,
        warm_corr=True,
        tolerance=0.00001,
        max_iters=100,
        ext_score_adjustment=0.5,
        method="log-lik",
        constant=None,
        threshold_constant=None,
    ):
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
        method : str, default 'log-lik'
            Priority vector extraction method.
        constant : float or None, default None
            Smoothing constant, passed to calibrate(); None resolves to 0.1.
        threshold_constant : float or None, default None
            Passed to calibrate() as the Stage 2 CPAT constant; None
            follows `constant` (0.03 when constant is None). See calibrate().
        """

        if not hasattr(self, "eigenvectors"):
            # BUG FIX: must call res_corr_analysis (not just fit_statistics) to set eigenvectors
            self.res_corr_analysis(
                warm_corr=warm_corr,
                tolerance=tolerance,
                max_iters=max_iters,
                ext_score_adjustment=ext_score_adjustment,
                method=method,
                constant=constant,
                threshold_constant=threshold_constant,
            )

        frames = [
            self.eigenvectors,
            self.eigenvalues,
            self.variance_explained,
            self.loadings,
        ]
        sheet_single = "Item residual analysis"
        sheet_multi = [
            "Eigenvectors",
            "Eigenvalues",
            "Variance explained",
            "Principal Component loadings",
        ]
        csv_suffixes = [
            "_eigenvectors",
            "_eigenvalues",
            "_variance_explained",
            "_principal_component_loadings",
        ]

        if format == "xlsx":
            if not filename.endswith(".xlsx"):
                filename += ".xlsx"
            with pd.ExcelWriter(filename, engine="openpyxl") as writer:
                if single:
                    row = 0
                    for frame in frames:
                        frame.round(dp).to_excel(
                            writer, sheet_name=sheet_single, startrow=row, startcol=0
                        )
                        row += frame.shape[0] + 2
                else:
                    for frame, sheet in zip(frames, sheet_multi):
                        frame.round(dp).to_excel(writer, sheet_name=sheet)
        else:
            if single:
                if not filename.endswith(".csv"):
                    filename += ".csv"
                with open(filename, "a") as f:
                    for frame in frames:
                        frame.round(dp).to_csv(f)
                        f.write("\n")
            else:
                if filename.endswith(".csv"):
                    filename = filename[:-4]
                for frame, suffix in zip(frames, csv_suffixes):
                    frame.round(dp).to_csv(f"{filename}{suffix}.csv")

    # ------------------------------------------------------------------
    # Class intervals (for ICC/CRC observed data overlay)
    # ------------------------------------------------------------------

    def class_intervals(self, items=None, no_of_classes=5):
        """
        Compute class interval mean person locations and mean observed total scores.

        Partitions persons into quantile-based person-location groups and computes
        mean person location and mean observed total score within each group.
        Used for observed-data overlays on TCC and ICC plots.
        Requires self.persons to exist.

        Parameters
        ----------
        items : str, list, or None, default None
            Item subset. None uses all items.
        no_of_classes : int, default 5
            Number of class intervals.

        Returns
        -------
        mean_person_locations : pandas.Series
            Mean person location within each class interval.
        obs : pandas.Series
            Mean observed total score within each class interval.
        """

        if isinstance(items, str) and items in ("all", "none"):
            items = None
        if items is None:
            items = self.responses.columns.tolist()

        class_groups = [f"class_{i + 1}" for i in range(no_of_classes)]
        df = self.responses[items].dropna(how="all")
        estimates = self.persons.loc[df.index]
        q = estimates.quantile(
            [(i + 1) / no_of_classes for i in range(no_of_classes - 1)]
        )

        mask_dict = {
            "class_1": estimates < q.values[0],
            f"class_{no_of_classes}": estimates >= q.values[-1],
            **{
                f"class_{i + 2}": (
                    (estimates >= q.values[i]) & (estimates < q.values[i + 1])
                )
                for i in range(no_of_classes - 2)
            },
        }
        mean_person_locations = pd.Series(
            {cg: estimates[mask_dict[cg]].mean() for cg in class_groups}
        )
        obs = pd.concat(
            {cg: pd.Series(df[mask_dict[cg]].mean().sum()) for cg in class_groups}
        )
        return mean_person_locations, obs

    def class_intervals_cats(self, person_locations, item=None, no_of_classes=5):
        """
        Compute class interval mean person locations and observed category proportions.

        Partitions persons into quantile-based person-location groups and computes the
        proportion of each response category within each group. When item=None,
        pools across all items using person location relative to each item's location.
        Used for observed-data overlays on CRC plots.

        Parameters
        ----------
        person_locations : pandas.Series
            Person location estimates indexed by person identifier.
        item : str or None, default None
            Item identifier. If None, pools across all items.
        no_of_classes : int, default 5
            Number of class intervals.

        Returns
        -------
        mean_person_locations : pandas.Series
            Mean person location within each class interval.
        obs_props : numpy.ndarray
            Shape (no_of_classes, max_score+1) with proportions of each
            response category in each class interval.
        """

        class_groups = [f"class_{i + 1}" for i in range(no_of_classes)]
        df = self.responses.copy()

        if item is None:
            # Use person location relative to each item's location
            person_location_df = pd.DataFrame(
                {
                    item_: person_locations - self.items[item_]
                    for item_ in self.responses.columns
                }
            ) * df.notna().astype(float).replace(0, np.nan)
            mask_scores = df.unstack()
            mask_person_locations = person_location_df.unstack()
        else:
            mask_scores = df[item].dropna()
            mask_person_locations = (person_locations - self.items[item]).reindex(mask_scores.index)
        q = mask_person_locations.quantile(
            [(i + 1) / no_of_classes for i in range(no_of_classes - 1)]
        )
        mask_dict = {
            "class_1": mask_person_locations < q.values[0],
            f"class_{no_of_classes}": mask_person_locations >= q.values[-1],
            **{
                f"class_{i + 2}": (
                    (mask_person_locations >= q.values[i]) & (mask_person_locations < q.values[i + 1])
                )
                for i in range(no_of_classes - 2)
            },
        }
        mean_person_locations = pd.Series(
            {cg: mask_person_locations[mask_dict[cg]].mean() for cg in class_groups}
        )
        obs_props = np.array(
            [
                [
                    (mask_scores[mask_dict[cg]] == cat).sum()
                    for cat in range(self.max_score + 1)
                ]
                for cg in class_groups
            ],
            dtype=float,
        )
        obs_props /= obs_props.sum(axis=1, keepdims=True)
        return mean_person_locations, obs_props

    def class_intervals_thresholds(self, item=None, no_of_classes=5):
        """
        Compute class interval data for threshold characteristic curves.

        For each threshold (adjacent category pair), dichotomises responses,
        partitions persons into quantile-based person-location groups, and computes the
        mean person location and observed proportion in the higher category within each
        group. When item=None, pools across all items.
        Auto-triggers person_estimates() if not yet run.

        Parameters
        ----------
        item : str or None, default None
            Item identifier. If None, pools across all items.
        no_of_classes : int, default 5
            Number of class intervals.

        Returns
        -------
        mean_person_locations : numpy.ndarray
            Shape (no_of_classes, max_score).
        obs_props : numpy.ndarray
            Shape (no_of_classes, max_score).
        """

        if not hasattr(self, "persons"):
            self.person_estimates(warm_corr=False)

        class_groups = [f"class_{i + 1}" for i in range(no_of_classes)]
        df = self.responses.copy()

        # Build person location DataFrame; subtract item location if not item-specific
        person_location_df = pd.DataFrame({it: self.persons for it in self.responses.columns})
        if item is None:
            for it in self.responses.columns:
                person_location_df[it] -= self.items[it]
        else:
            df = df[item]
            person_location_df = person_location_df[item]

        def make_masks(estimates):
            q = estimates.quantile(
                [(i + 1) / no_of_classes for i in range(no_of_classes - 1)]
            )
            md = {
                "class_1": estimates < q.values[0],
                f"class_{no_of_classes}": estimates >= q.values[-1],
                **{
                    f"class_{i + 2}": (
                        (estimates >= q.values[i]) & (estimates < q.values[i + 1])
                    )
                    for i in range(no_of_classes - 2)
                },
            }
            return {cg: md[cg][md[cg]].index for cg in class_groups}

        mean_person_locations, obs_props = [], []
        for t in range(self.max_score):
            cond_df = df[df.isin([t, t + 1])] - t
            cond_mask = cond_df.notna().astype(float).replace(0, np.nan)
            cond_person_locations = person_location_df * cond_mask

            if item is None:
                obs_df = pd.DataFrame(
                    {"person_location": cond_person_locations.stack(), "score": cond_df.stack()}
                ).droplevel(level=1)
            else:
                obs_df = pd.DataFrame({"person_location": cond_person_locations, "score": cond_df})

            masks = make_masks(obs_df["person_location"])
            mean_person_locations.append(
                [obs_df.loc[masks[cg]]["person_location"].mean() for cg in class_groups]
            )
            obs_props.append(
                [obs_df.loc[masks[cg]]["score"].mean() for cg in class_groups]
            )

        return np.array(mean_person_locations).T, np.array(obs_props).T

    def _label_height_frac(self, font, axis_font_size, figsize):
        """
        Fraction of the axes' own pixel height that one line of label
        text occupies, at the given font/size/figure size -- measured
        directly via a probe render (not guessed), so callers can
        convert it into an exact data-unit label height for whatever
        y-range they end up using (that height scales linearly with the
        y-range, since the axes' pixel height itself doesn't depend on
        the data plotted). Used to size ymax and place stacked
        central_location labels with a small, guaranteed gap rather
        than a fixed fraction of y_max (see feedback_plot_layout_rigor
        -- measure, don't guess).
        """
        probe_fig = plt.figure(figsize=figsize)
        probe_ax = probe_fig.add_subplot(111)
        probe_txt = probe_ax.text(
            0, 0, "Item_1: 0.000", fontsize=axis_font_size, fontfamily=font
        )
        probe_fig.canvas.draw()
        renderer = probe_fig.canvas.get_renderer()
        label_h_px = probe_txt.get_window_extent(renderer=renderer).height
        ax_h_px = probe_ax.get_window_extent(renderer=renderer).height
        plt.close(probe_fig)
        return label_h_px / ax_h_px

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    def plot_data(
        self,
        x_data,
        y_data,
        items=None,
        curve_labels=None,
        legend_loc=None,
        obs=None,
        obs_curve_index=None,
        marker_border=True,
        line_border=False,
        x_obs_data=np.array([]),
        y_obs_data=np.array([]),
        thresh_lines=False,
        central_location=False,
        central_location_fit=False,
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
        graph_title="",
        y_label="",
        plot_style="white",
        palette="colorblind multi",
        black=False,
        figsize=(8, 6),
        font="Times New Roman",
        title_font_size=15,
        axis_font_size=12,
        labelsize=12,
        tex=True,
        plot_density=300,
        filename=None,
        file_format="png",
    ):
        """
        Core plotting engine for all RSM item and test characteristic curves.

        Renders curves against a person-location x-axis with optional observed overlays,
        threshold lines, central difference lines, score lines, information lines,
        and CSEM lines. Called internally by icc(), crcs(), threshold_ccs(),
        iic(), tcc(), test_info(), and test_csem().

        Parameters
        ----------
        x_data : array-like
            X-axis values (typically person-location grid -20 to 20).
        y_data : numpy.ndarray
            2-D array shape (len(x_data), n_curves).
        items : str, list, or None
            Item(s) being plotted.
        curve_labels : list of str, or None, default None
            One label per curve (matching y_data's columns), shown in a
            legend. None (default) leaves curves unlabelled, as before.
            icc() passes item names here when plotting more than one
            item at once.
        obs : bool, list, or None
            Controls observed data overlay.
        obs_curve_index : list of int, or None, default None
            Only relevant when y_obs_data has more than one column
            (multiple observed series). Maps each column position to the
            curve index (in y_data) it belongs to, so its marker is
            coloured to match that curve -- needed whenever the observed
            series have been subsetted/reordered relative to the full
            curve set (crcs()'s obs= is a list of category indices, not
            necessarily 0..n in order). None means the identity mapping
            (position i belongs to curve i), correct whenever the two
            already line up, e.g. icc()'s own multi-item obs columns.
        marker_border : bool, default True
            If True, observed-data markers get a black edge (matching
            plot_anchor_selection's own marker style). If False, markers
            are drawn with no edge.
        line_border : bool, default False
            If True, each curve line gets a thin contrasting stroke
            (white on the darker plot_style schemes -- see
            _DARK_BACKGROUND_STYLES -- black otherwise), so it stays
            legible against a background/gridline colour close to its
            own. Off by default; the 'colorblind multi' palette's own
            black entry is separately swapped for white on those same
            dark schemes regardless of this flag.
        x_obs_data, y_obs_data : array-like
            Observed data point coordinates.
        thresh_lines : bool, default False
            Draw vertical lines at absolute threshold locations.
        central_location : bool, default False
            Draw a line at the item central location.
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
            Used for score line person-location lookups.
        cat_highlight : int or None
            Category to shade blue.
        graph_title, y_label : str
            Plot title and y-axis label.
        plot_style : str, default 'white'
            One of self._PLOT_STYLE_RC's keys: 'white', 'dark', 'black',
            'parchment', 'minimal', 'print', 'solarized-light',
            'solarized-dark', 'slate', 'blueprint', 'newsprint', or
            'chalkboard'.
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
        self._apply_plot_style(plot_style)

        palette_dict = {
            "dark blue": ["dark", "royalblue"],
            "light blue": ["light", "cornflowerblue"],
            "dark red": ["dark", "firebrick"],
            "light red": ["light", "indianred"],
            "dark green": ["dark", "forestgreen"],
            "light green": ["light", "mediumseagreen"],
            "dark grey": ["dark", "dimgrey"],
            "light grey": ["light", "darkgrey"],
            "dark multi": ["dark", "dark"],
            "light multi": ["light", "muted"],
            "colorblind multi": ["dark", "colorblind"],
        }

        shade, base_color = palette_dict[palette]
        if palette == "colorblind multi":
            # Okabe & Ito (2008) -- colour-vision-deficiency-safe
            # qualitative palette, the scientific-publishing standard.
            # The 8th (black) entry is swapped for white on dark
            # backgrounds, where it would otherwise be invisible.
            color_map = [
                "#E69F00", "#56B4E9", "#009E73", "#F0E442",
                "#0072B2", "#D55E00", "#CC79A7",
                "#FFFFFF" if plot_style in self._DARK_BACKGROUND_STYLES else "#000000",
            ]
        elif shade == "dark":
            color_map = (
                sns.color_palette("dark", as_cmap=True)
                if palette == "dark multi"
                else sns.dark_palette(base_color, reverse=True, as_cmap=True)
            )
        else:
            color_map = (
                sns.color_palette("muted", as_cmap=True)
                if palette == "light multi"
                else sns.light_palette(base_color, reverse=True, as_cmap=True)
            )

        with plt.rc_context({"font.family": font, "font.size": axis_font_size}):
            graph, ax = plt.subplots(figsize=figsize)
            no_of_plots = y_data.shape[1]
            cNorm = colors.Normalize(vmin=0, vmax=no_of_plots + 2)

            if "multi" not in palette:
                scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=color_map)

            # A thin contrasting stroke around each curve line, so it stays
            # legible on a background/gridline colour close to its own --
            # white on the darker plot_style schemes, black otherwise.
            line_fx = (
                [path_effects.withStroke(
                    linewidth=2.5,
                    foreground="white" if plot_style in self._DARK_BACKGROUND_STYLES else "black",
                )]
                if line_border
                else None
            )

            for i in range(no_of_plots):
                col = (
                    "black"
                    if black
                    else (
                        scalarMap.to_rgba(i) if "multi" not in palette else color_map[i]
                    )
                )
                label = curve_labels[i] if curve_labels is not None else i + 1
                ax.plot(
                    x_data, y_data[:, i], "", color=col, label=label,
                    path_effects=line_fx,
                )

            if curve_labels is not None:
                ax.legend(loc=legend_loc if legend_loc is not None else "best")

            if obs is not None:
                x_is_series = isinstance(x_obs_data, pd.Series)
                if np.ndim(y_obs_data) == 1:
                    col = (
                        scalarMap.to_rgba(0) if "multi" not in palette else color_map[0]
                    )
                    ax.scatter(
                        x_obs_data, y_obs_data, color=col, s=40, alpha=0.7,
                        edgecolors="k" if marker_border else "none", zorder=3,
                    )
                else:
                    try:
                        n_obs = y_obs_data.shape[1]
                        for j in range(n_obs):
                            k = obs_curve_index[j] if obs_curve_index is not None else j
                            col = (
                                scalarMap.to_rgba(k)
                                if "multi" not in palette
                                else color_map[k]
                            )
                            xd = x_obs_data if x_is_series else x_obs_data[:, j]
                            ax.scatter(
                                xd, y_obs_data[:, j], color=col, s=40, alpha=0.7,
                                edgecolors="k" if marker_border else "none", zorder=3,
                            )
                    except Exception:
                        pass

            if thresh_lines:
                for t in range(self.max_score):
                    xval = (
                        self.thresholds[t + 1]
                        if items is None
                        else self.thresholds[t + 1] + self.items.loc[items]
                    )
                    ax.axvline(x=xval, color="black", linestyle="--")
                    # Staggered onto two rows near the bottom so that any
                    # two adjacent thresholds (the only ones close enough
                    # in x to clash) never share a row.
                    label_y = y_max * (0.05 if t % 2 == 0 else 0.11)
                    ax.text(
                        xval + (x_max - x_min) / 100,
                        label_y,
                        str(t + 1),
                        color="black",
                        va="bottom",
                        ha="left",
                    )

            if central_location:
                # items may be a list (one curve per item) -- draw each
                # item's own central location, always plain darkred
                # (colour isn't needed to disambiguate since each line
                # is labelled with its own item name).
                items_multi = isinstance(items, list)
                items_list = items if items_multi else [items]
                n_labels = len(items_list)
                if central_location_fit and items_multi and n_labels > 0:
                    # "Fit" style (iic): the curve's own peak sits right
                    # at each item's central location, so labels are
                    # stacked with a probe-measured height above that
                    # peak -- ymax is pre-widened by the caller to fit
                    # them exactly (measure, don't guess -- see
                    # feedback_plot_layout_rigor).
                    k = self._label_height_frac(font, axis_font_size, figsize)
                    label_h_data = k * y_max
                    gap = 0.3 * label_h_data
                    curve_peak = float(np.nanmax(y_data))
                for idx, it in enumerate(items_list):
                    xval = 0 if it is None else self.items.loc[it]
                    ax.axvline(x=xval, color="darkred", linestyle="--")
                    label = (
                        f"{it}: {round(xval, 2)}" if items_multi else str(round(xval, 2))
                    )
                    if central_location_fit and items_multi:
                        label_y = curve_peak + gap + label_h_data * (n_labels - idx - 0.5)
                        label_x = xval - (x_max - x_min) / 100
                        ha = "right"
                    elif items_multi:
                        # "Top" style (icc): the curve is an ogive, near
                        # its ceiling by the time it nears max_score, so
                        # labels just stack near the top of the fixed
                        # y_max instead -- placed to the *left* of each
                        # line (curves rise left-to-right, so that side
                        # stays clear of the curve even close to the
                        # ceiling) rather than boosting y_max past its
                        # own meaningful value.
                        label_y = y_max * 0.95 - idx * y_max * 0.05
                        label_x = xval - (x_max - x_min) / 100
                        ha = "right"
                    else:
                        label_y = y_max * 0.95
                        label_x = xval + (x_max - x_min) / 100
                        ha = "left"
                        label = f"Central location: {round(xval, 2)}"
                    ax.text(
                        label_x,
                        label_y,
                        label,
                        color="black",
                        va="center",
                        ha=ha,
                    )

            if score_lines_item[1] is not None:
                # score_lines_item[0] may be a list (one curve per item)
                # -- draw each item's own score lines in that item's own
                # curve colour. Single-item calls keep the original plain
                # black lines.
                items_arg = score_lines_item[0]
                items_multi = isinstance(items_arg, list)
                items_list = items_arg if items_multi else [items_arg]
                if all(s > 0 for s in score_lines_item[1]) and all(
                    s < self.max_score for s in score_lines_item[1]
                ):
                    for idx, it in enumerate(items_list):
                        colorVal = (
                            "black"
                            if black or not items_multi
                            else (
                                scalarMap.to_rgba(idx)
                                if "multi" not in palette
                                else color_map[idx]
                            )
                        )
                        for s in score_lines_item[1]:
                            estimate = self.score_lookup(s, items=[it], warm_corr=False)
                            ax.vlines(
                                x=estimate,
                                ymin=-100,
                                ymax=s,
                                color=colorVal,
                                linestyles="dashed",
                            )
                            ax.hlines(
                                y=s,
                                xmin=-100,
                                xmax=estimate,
                                color=colorVal,
                                linestyles="dashed",
                            )
                            if score_labels:
                                # Stagger each item's estimate label a bit
                                # higher up than the last, so nearby
                                # curves' labels don't overwrite each
                                # other -- single-item calls keep the
                                # original fixed height.
                                label_y = y_max / 50 + (
                                    idx * y_max * 0.05 if items_multi else 0
                                )
                                ax.text(
                                    estimate + (x_max - x_min) / 100,
                                    label_y,
                                    str(round(estimate, 2)),
                                    color=colorVal,
                                )
                                ax.text(
                                    x_min + (x_max - x_min) / 100, s + y_max / 50, str(s),
                                    color=colorVal,
                                )
                else:
                    warnings.warn(
                        "Invalid score for score line: value must be "
                        "strictly between 0 and the item maximum score.",
                        UserWarning,
                        stacklevel=2,
                    )

            if score_lines_test is not None:
                item_keys = (
                    self.responses.columns
                    if items is None
                    else ([items] if isinstance(items, str) else items)
                )
                n_items = len(item_keys)
                if all(s > 0 for s in score_lines_test) and all(
                    s < self.max_score * n_items for s in score_lines_test
                ):
                    for s in score_lines_test:
                        estimate = self.score_lookup(
                            s, items=list(item_keys), warm_corr=warm
                        )
                        ax.vlines(
                            x=estimate,
                            ymin=-100,
                            ymax=s,
                            color="black",
                            linestyles="dashed",
                        )
                        ax.hlines(
                            y=s,
                            xmin=-100,
                            xmax=estimate,
                            color="black",
                            linestyles="dashed",
                        )
                        if score_labels:
                            ax.text(
                                estimate + (x_max - x_min) / 100,
                                y_max / 50,
                                str(round(estimate, 2)),
                            )
                            ax.text(
                                x_min + (x_max - x_min) / 100, s + y_max / 50, str(s)
                            )
                else:
                    warnings.warn(
                        "Invalid score for score line: value must be "
                        "strictly between 0 and the test maximum score.",
                        UserWarning,
                        stacklevel=2,
                    )

            if point_info_lines_item[1] is not None:
                # items may be a list (one curve per item) -- for a
                # given location, every item's own information value is
                # genuinely different (that's the whole point of
                # comparing them), so each gets its own "item: info"
                # label at its own natural height, nudged sideways per
                # item so close values don't collide. Single-item calls
                # keep the original separate location/info label pair.
                items_arg = point_info_lines_item[0]
                items_multi = isinstance(items_arg, list)
                items_list = items_arg if items_multi else [items_arg]
                for idx, it in enumerate(items_list):
                    colorVal = (
                        "black"
                        if black or not items_multi
                        else (
                            scalarMap.to_rgba(idx)
                            if "multi" not in palette
                            else color_map[idx]
                        )
                    )
                    for estimate in point_info_lines_item[1]:
                        info = self.variance(estimate, self.items[it], self.thresholds)
                        ax.vlines(
                            x=estimate, ymin=-100, ymax=info, color=colorVal,
                            linestyles="dashed",
                        )
                        ax.hlines(
                            y=info, xmin=-100, xmax=estimate, color=colorVal,
                            linestyles="dashed",
                        )
                        if score_labels:
                            if items_multi:
                                label_x = (
                                    x_min
                                    + (x_max - x_min) / 100
                                    + idx * (x_max - x_min) * 0.03
                                )
                                ax.text(
                                    label_x, info + y_max / 50,
                                    f"{it}: {round(info, 3)}", color=colorVal,
                                )
                            else:
                                ax.text(
                                    estimate + (x_max - x_min) / 100,
                                    y_max / 50,
                                    str(round(estimate, 2)),
                                )
                                ax.text(
                                    x_min + (x_max - x_min) / 100,
                                    info + y_max / 50,
                                    str(round(info, 3)),
                                )

            if point_info_lines_test is not None:
                item_keys = self.responses.columns if items is None else items
                for estimate in point_info_lines_test:
                    info = sum(
                        self.variance(estimate, self.items[it], self.thresholds)
                        for it in item_keys
                    )
                    ax.vlines(
                        x=estimate,
                        ymin=-100,
                        ymax=info,
                        color="black",
                        linestyles="dashed",
                    )
                    ax.hlines(
                        y=info,
                        xmin=-100,
                        xmax=estimate,
                        color="black",
                        linestyles="dashed",
                    )
                    if score_labels:
                        ax.text(
                            estimate + (x_max - x_min) / 100,
                            y_max / 50,
                            str(round(estimate, 2)),
                        )
                        ax.text(
                            x_min + (x_max - x_min) / 100,
                            info + y_max / 50,
                            str(round(info, 3)),
                        )

            if point_csem_lines is not None:
                item_keys = self.responses.columns if items is None else items
                for estimate in point_csem_lines:
                    info = sum(
                        self.variance(estimate, self.items[it], self.thresholds)
                        for it in item_keys
                    )
                    csem = 1.0 / (info**0.5)
                    ax.vlines(
                        x=estimate,
                        ymin=-100,
                        ymax=csem,
                        color="black",
                        linestyles="dashed",
                    )
                    ax.hlines(
                        y=csem,
                        xmin=-100,
                        xmax=estimate,
                        color="black",
                        linestyles="dashed",
                    )
                    if score_labels:
                        ax.text(
                            estimate + (x_max - x_min) / 100,
                            y_max / 50,
                            str(round(estimate, 2)),
                        )
                        ax.text(
                            x_min + (x_max - x_min) / 100,
                            csem + y_max / 50,
                            str(round(csem, 3)),
                        )

            if items is not None and cat_highlight in range(self.max_score + 1):
                if cat_highlight == 0:
                    ax.axvspan(
                        -100,
                        self.items[items] + self.thresholds[1],
                        facecolor="blue",
                        alpha=0.2,
                    )
                elif cat_highlight == self.max_score:
                    ax.axvspan(
                        self.items[items] + self.thresholds[self.max_score],
                        100,
                        facecolor="blue",
                        alpha=0.2,
                    )
                else:
                    lo = self.items[items] + self.thresholds[cat_highlight]
                    hi = self.items[items] + self.thresholds[cat_highlight + 1]
                    if hi > lo:
                        ax.axvspan(lo, hi, facecolor="blue", alpha=0.2)

            if y_max <= 0:
                y_max = float(y_data.max()) * 1.1

            ax.set_xlim(x_min, x_max)
            ax.set_ylim(0, y_max)
            ax.set_xlabel("Person location", fontsize=axis_font_size, fontweight="bold")
            ax.set_ylabel(y_label, fontsize=axis_font_size, fontweight="bold")
            ax.set_title(graph_title, fontsize=title_font_size, fontweight="bold")
            ax.grid(True)
            ax.tick_params(axis="x", labelsize=labelsize)
            ax.tick_params(axis="y", labelsize=labelsize)

            if filename is not None:
                graph.savefig(f"{filename}.{file_format}", dpi=plot_density)

            plt.close(graph)

        return graph

    def icc(
        self,
        item,
        obs=False,
        no_of_classes=5,
        title=None,
        thresh_lines=False,
        central_location=False,
        score_lines=None,
        score_labels=False,
        cat_highlight=None,
        xmin=-5,
        xmax=5,
        plot_style="white",
        palette="colorblind multi",
        black=False,
        marker_border=True,
        line_border=False,
        font="Times New Roman",
        title_font_size=15,
        axis_font_size=12,
        labelsize=12,
        filename=None,
        file_format="png",
        dpi=300,
    ):
        """
        Plot the Item Characteristic Curve (ICC) for one item, or several
        overlaid on the same axes.

        Displays modelled expected score as a function of person location. Optionally
        overlays observed class-interval mean scores -- single-item only, see item below.

        Parameters
        ----------
        item : str or list of str
            Item identifier(s). A single name draws one curve, as before.
            A list overlays one curve per item (in the given order), each
            in its own colour with a legend keyed by item name.
            thresh_lines/cat_highlight only make sense for one item's
            own location, so they just silently no-op with several
            plotted at once rather than erroring. obs, central_location,
            and score_lines all work fine with a list -- each item's own
            point(s)/line(s) are drawn in that item's own curve colour,
            so they stay distinguishable.
        obs : bool, default False
            If True, overlays observed class-interval mean scores. Works
            with a list of items too -- one column of observed points
            per item, in that item's own curve colour (class intervals
            are the same person-location quantile groups for every item,
            computed from self.persons, so the same x-axis positions are
            shared).
        no_of_classes : int, default 5
            Number of class intervals.
        title : str or None, default None
            Plot title.
        thresh_lines : bool, default False
            Draw vertical lines at absolute threshold locations (tau_k + delta_i).
            Single item only -- silently ignored if item is a list.
        central_location : bool, default False
            Draw a line at the item central location, labelled with the
            value (and item name, when item is a list -- staggered
            downward from the top, one step per item, so nearby curves'
            labels don't overwrite each other). Works with a list of
            items too.
        score_lines : list or None, default None
            Raw scores at which to draw reference lines. Works with a
            list of items too.
        score_labels : bool, default False
            Annotate score line intersections.
        cat_highlight : int or None, default None
            Category to shade. Single item only -- silently ignored if
            item is a list.
        xmin, xmax : float
            Person-location axis limits.
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
        multi = isinstance(item, list)
        if multi:
            # thresh_lines/cat_highlight each only make sense for a single
            # item's own location -- silently no-op with several items
            # plotted at once rather than erroring, since they'd just be
            # visual clutter with no obvious single "right" item anyway.
            thresh_lines = False
            cat_highlight = None

        # BUG FIX: typo 'person_abiliites'
        if obs and not hasattr(self, "persons"):
            self.person_estimates(warm_corr=False)

        xobsdata = yobsdata = np.array(np.nan)
        if obs:
            if multi:
                # Same person-location class intervals for every item
                # (quantile groups come from self.persons, the overall
                # ability estimate, not an item-specific one -- so they
                # line up across items with complete data; the first
                # item's own x values are reused for the rest). One
                # observed mean-score column per item, in that item's
                # own curve colour.
                xobsdata = None
                yobs_cols = []
                for it in item:
                    mpl, om = self.class_intervals(items=[it], no_of_classes=no_of_classes)
                    if xobsdata is None:
                        xobsdata = pd.Series(mpl)
                    yobs_cols.append(np.array(om))
                yobsdata = np.column_stack(yobs_cols)
            else:
                mean_person_locations, obs_means = self.class_intervals(
                    items=item, no_of_classes=no_of_classes
                )
                xobsdata = pd.Series(mean_person_locations)
                yobsdata = np.array(obs_means).reshape(-1, 1)

        estimates = np.arange(-20, 20, 0.1)
        if multi:
            y = np.column_stack(
                [
                    [self.exp_score(a, self.items[it], self.thresholds) for a in estimates]
                    for it in item
                ]
            )
        else:
            y = np.array(
                [self.exp_score(a, self.items[item], self.thresholds) for a in estimates]
            ).reshape(-1, 1)

        return self.plot_data(
            x_data=estimates,
            y_data=y,
            x_obs_data=xobsdata,
            y_obs_data=yobsdata,
            x_min=xmin,
            x_max=xmax,
            y_max=self.max_score,
            items=item,
            curve_labels=item if multi else None,
            legend_loc="upper left" if score_lines is not None else "lower right",
            graph_title=title or "",
            y_label="Expected score",
            obs=obs,
            marker_border=marker_border,
            line_border=line_border,
            thresh_lines=thresh_lines,
            central_location=central_location,
            score_lines_item=[item, score_lines],
            score_labels=score_labels,
            plot_style=plot_style,
            palette=palette,
            black=black,
            font=font,
            cat_highlight=cat_highlight,
            title_font_size=title_font_size,
            axis_font_size=axis_font_size,
            labelsize=labelsize,
            filename=filename,
            plot_density=dpi,
            file_format=file_format,
        )

    def crcs(
        self,
        item=None,
        obs=None,
        no_of_classes=5,
        title=None,
        thresh_lines=False,
        central_location=False,
        cat_highlight=None,
        xmin=-5,
        xmax=5,
        plot_style="white",
        palette="colorblind multi",
        black=False,
        marker_border=True,
        line_border=False,
        font="Times New Roman",
        title_font_size=15,
        axis_font_size=12,
        labelsize=12,
        filename=None,
        file_format="png",
        dpi=300,
    ):
        """
        Plot Category Response Curves (CRCs) for a single item.

        Displays the probability of each response category as a function of
        person location using the RSM centred parameterisation. Optionally overlays
        observed category proportions.

        Parameters
        ----------
        item : str or None, default None
            Item identifier. If None, uses zero location.
        obs : list, 'all', or None, default None
            Observed overlay: 'all', list of category indices, or None.
        no_of_classes : int, default 5
            Number of class intervals.
        title : str or None, default None
            Plot title.
        thresh_lines : bool, default False
            Draw vertical lines at absolute threshold locations.
        central_location : bool, default False
            Draw a line at the item central location.
        cat_highlight : int or None, default None
            Category to shade.
        xmin, xmax : float
            Person-location axis limits.
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
        if item == "none":
            item = None
        # BUG FIX: typo 'person_abiliites'
        if obs is not None and not hasattr(self, "persons"):
            self.person_estimates(warm_corr=False)

        xobsdata = yobsdata = np.array(np.nan)
        if obs is not None:
            xobsdata, yobsdata = self.class_intervals_cats(
                self.persons, item=item, no_of_classes=no_of_classes
            )
            if isinstance(obs, str) and obs == "all":
                obs = np.arange(self.max_score + 1)
            if not all(c in np.arange(self.max_score + 1) for c in obs):
                warnings.warn(
                    "Invalid 'obs' value. Valid values are None, 'all', "
                    "or a list of category indices.",
                    UserWarning,
                    stacklevel=2,
                )
                return
            yobsdata = yobsdata[:, obs]

        estimates = np.arange(-20, 20, 0.1)
        diff = 0 if item is None else self.items[item]
        y = np.array(
            [
                [
                    self.cat_prob(a, diff, cat, self.thresholds)
                    for cat in range(self.max_score + 1)
                ]
                for a in estimates
            ]
        )

        return self.plot_data(
            x_data=estimates,
            y_data=y,
            x_min=xmin,
            x_max=xmax,
            y_max=1,
            x_obs_data=xobsdata,
            y_obs_data=yobsdata,
            items=item,
            curve_labels=[f"Category {c}" for c in range(self.max_score + 1)],
            graph_title=title or "",
            y_label="Probability",
            obs=obs,
            obs_curve_index=obs if obs is not None else None,
            marker_border=marker_border,
            line_border=line_border,
            thresh_lines=thresh_lines,
            central_location=central_location,
            cat_highlight=cat_highlight,
            plot_style=plot_style,
            palette=palette,
            black=black,
            font=font,
            title_font_size=title_font_size,
            axis_font_size=axis_font_size,
            labelsize=labelsize,
            filename=filename,
            plot_density=dpi,
            file_format=file_format,
        )

    def threshold_ccs(
        self,
        item=None,
        obs=None,
        no_of_classes=5,
        title=None,
        thresh_lines=False,
        central_location=False,
        cat_highlight=None,
        xmin=-5,
        xmax=5,
        plot_style="white",
        palette="colorblind multi",
        black=False,
        marker_border=True,
        line_border=False,
        font="Times New Roman",
        title_font_size=15,
        axis_font_size=12,
        labelsize=12,
        filename=None,
        file_format="png",
        dpi=300,
    ):
        """
        Plot Threshold Characteristic Curves (TCCs).

        Displays the probability of scoring in the higher of two adjacent
        categories at each shared threshold. When item=None, plots thresholds
        at their shared locations without item location offset.

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
        central_location : bool, default False
            Draw a line at the item central location.
        cat_highlight : int or None, default None
            Threshold category to shade.
        xmin, xmax : float
            Person-location axis limits.
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
        if item == "none":
            item = None
        # BUG FIX: typo 'person_abiliites'
        if obs is not None and not hasattr(self, "persons"):
            self.person_estimates(warm_corr=False)

        xobsdata = yobsdata = np.array(np.nan)
        obs_curve_index = None
        if obs is not None:
            mean_person_locations, obs_props = self.class_intervals_thresholds(
                item=item, no_of_classes=no_of_classes
            )
            xobsdata, yobsdata = mean_person_locations, obs_props
            if obs != "all":
                if not all(c in np.arange(self.max_score) + 1 for c in obs):
                    warnings.warn(
                        "Invalid 'obs' value. Valid values are None, 'all', "
                        "or a list of threshold numbers.",
                        UserWarning,
                        stacklevel=2,
                    )
                    return
                obs_idx = [o - 1 for o in obs]
                xobsdata = xobsdata[:, obs_idx]
                yobsdata = yobsdata[:, obs_idx]
                obs_curve_index = obs_idx

        estimates = np.arange(-20, 20, 0.1)
        # Absolute threshold locations: tau_k (+ item location if item-specific)
        abs_thresh = (
            self.thresholds if item is None else self.thresholds + self.items[item]
        )
        y = np.array(
            [[1.0 / (1.0 + np.exp(thr - a)) for thr in abs_thresh] for a in estimates]
        )

        return self.plot_data(
            x_data=estimates,
            y_data=y,
            y_max=1,
            x_min=xmin,
            x_max=xmax,
            items=item,
            curve_labels=[f"Threshold {t + 1}" for t in range(self.max_score)],
            obs=obs,
            obs_curve_index=obs_curve_index,
            marker_border=marker_border,
            line_border=line_border,
            x_obs_data=xobsdata,
            y_obs_data=yobsdata,
            graph_title=title or "",
            y_label="Probability",
            thresh_lines=thresh_lines,
            central_location=central_location,
            cat_highlight=cat_highlight,
            plot_style=plot_style,
            palette=palette,
            black=black,
            font=font,
            title_font_size=title_font_size,
            axis_font_size=axis_font_size,
            labelsize=labelsize,
            filename=filename,
            file_format=file_format,
            plot_density=dpi,
        )

    def iic(
        self,
        item,
        ymax=None,
        thresh_lines=False,
        central_location=False,
        point_info_lines=None,
        point_info_labels=False,
        cat_highlight=None,
        title=None,
        xmin=-5,
        xmax=5,
        plot_style="white",
        palette="colorblind multi",
        black=False,
        font="Times New Roman",
        title_font_size=15,
        axis_font_size=12,
        labelsize=12,
        filename=None,
        file_format="png",
        dpi=300,
    ):
        """
        Plot the Item Information Curve (IIC) for one item, or several
        overlaid on the same axes.

        Displays Fisher information as a function of person location.

        Parameters
        ----------
        item : str or list of str
            Item identifier(s). A single name draws one curve, as before.
            A list overlays one curve per item (in the given order), each
            in its own colour with a legend keyed by item name.
            thresh_lines/cat_highlight only make sense for one item's
            own location, so they just silently no-op with several
            plotted at once rather than erroring. central_location and
            point_info_lines both work fine with a list -- central_location
            draws each item's own line, labelled with its value (and
            name); point_info_lines draws every item's own information
            value at each requested location, labelled "item: info",
            since that comparison across items at a shared location is
            the point.
        ymax : float or None, default None
            Upper y-axis limit. Auto-scaled if None.
        thresh_lines : bool, default False
            Draw vertical lines at absolute threshold locations. Single
            item only -- silently ignored if item is a list.
        central_location : bool, default False
            Draw a line at the item central location. Works with a list
            of items too.
        point_info_lines : list or None, default None
            Person-location values at which to draw information reference lines.
            Works with a list of items too.
        point_info_labels : bool, default False
            Annotate information line intersections.
        cat_highlight : int or None, default None
            Category to shade. Single item only -- silently ignored if
            item is a list.
        title : str or None, default None
            Plot title.
        xmin, xmax : float
            Person-location axis limits.
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
        multi = isinstance(item, list)
        if multi:
            # thresh_lines/cat_highlight each only make sense for a
            # single item's own location -- silently no-op with several
            # items plotted at once rather than erroring, since they'd
            # just be visual clutter with no obvious single "right" item
            # anyway.
            thresh_lines = False
            cat_highlight = None

        estimates = np.arange(-20, 20, 0.1)
        if multi:
            y = np.column_stack(
                [
                    [self.variance(a, self.items[it], self.thresholds) for a in estimates]
                    for it in item
                ]
            )
        else:
            y = np.array(
                [self.variance(a, self.items[item], self.thresholds) for a in estimates]
            ).reshape(-1, 1)
        if ymax is None:
            ymax = float(y.max()) * 1.1
            if central_location and multi and len(item) > 0:
                # central_location's labels stack above the curve's own
                # peak (which sits right at each item's own central
                # location -- exactly where those labels are), so widen
                # ymax by exactly the measured label-stack height plus a
                # small gap, rather than guessing a fixed fraction (see
                # _label_height_frac / feedback_plot_layout_rigor --
                # measure, don't guess). Labels are then placed using
                # this same y.max()/k/gap_frac relationship in plot_data,
                # so the fit is exact, not approximate.
                n_labels = len(item)
                k = self._label_height_frac(font, axis_font_size, (8, 6))
                gap_frac = 0.3
                denom = max(1 - k * (n_labels + 2 * gap_frac), 0.1)
                ymax = max(ymax, float(y.max()) / denom)

        return self.plot_data(
            x_data=estimates,
            y_data=y,
            x_min=xmin,
            x_max=xmax,
            y_max=ymax,
            items=item,
            curve_labels=item if multi else None,
            thresh_lines=thresh_lines,
            central_location=central_location,
            central_location_fit=True,
            point_info_lines_item=[item, point_info_lines],
            score_labels=point_info_labels,
            cat_highlight=cat_highlight,
            graph_title=title or "",
            y_label="Fisher information",
            plot_style=plot_style,
            palette=palette,
            black=black,
            font=font,
            title_font_size=title_font_size,
            axis_font_size=axis_font_size,
            labelsize=labelsize,
            filename=filename,
            plot_density=dpi,
            file_format=file_format,
        )

    def tcc(
        self,
        items=None,
        obs=False,
        no_of_classes=5,
        title=None,
        score_lines=None,
        score_labels=False,
        xmin=-5,
        xmax=5,
        plot_style="white",
        palette="colorblind multi",
        black=False,
        marker_border=True,
        line_border=False,
        font="Times New Roman",
        title_font_size=15,
        axis_font_size=12,
        labelsize=12,
        filename=None,
        file_format="png",
        dpi=300,
    ):
        """
        Plot the Test Characteristic Curve (TCC).

        Displays expected total score as a function of person location. Optionally
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
            Person-location axis limits.
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
        if isinstance(items, str) and items in ("all", "none"):
            items = None
        elif isinstance(items, str):
            items = [items]

        # BUG FIX: typo 'person_abiliites'
        if obs and not hasattr(self, "persons"):
            self.person_estimates(warm_corr=False)

        xobsdata = yobsdata = np.array(np.nan)
        if obs:
            mean_person_locations, obs_means = self.class_intervals(
                items=items, no_of_classes=no_of_classes
            )
            xobsdata = mean_person_locations
            yobsdata = np.array(obs_means).reshape(no_of_classes, 1)

        estimates = np.arange(-20, 20, 0.1)
        item_keys = list(self.responses.columns) if items is None else items
        y = np.array(
            [
                sum(
                    self.exp_score(a, self.items[it], self.thresholds)
                    for it in item_keys
                )
                for a in estimates
            ]
        ).reshape(-1, 1)
        y_max = self.max_score * len(item_keys)

        return self.plot_data(
            x_data=estimates,
            y_data=y,
            items=items,
            x_obs_data=xobsdata,
            y_obs_data=yobsdata,
            x_min=xmin,
            x_max=xmax,
            y_max=y_max,
            score_lines_test=score_lines,
            score_labels=score_labels,
            graph_title=title or "",
            y_label="Expected score",
            obs=obs,
            marker_border=marker_border,
            line_border=line_border,
            plot_style=plot_style,
            palette=palette,
            black=black,
            font=font,
            title_font_size=title_font_size,
            axis_font_size=axis_font_size,
            labelsize=labelsize,
            filename=filename,
            plot_density=dpi,
            file_format=file_format,
        )

    def test_info(
        self,
        items=None,
        point_info_lines=None,
        point_info_labels=False,
        xmin=-5,
        xmax=5,
        ymax=None,
        title=None,
        plot_style="white",
        palette="colorblind multi",
        black=False,
        font="Times New Roman",
        title_font_size=15,
        axis_font_size=12,
        labelsize=12,
        filename=None,
        file_format="png",
        dpi=300,
    ):
        """
        Plot the Test Information Curve.

        Displays sum of item Fisher information values as a function of person location.

        Parameters
        ----------
        items : str, list, or None, default None
            Item subset. None uses all items.
        point_info_lines : list or None, default None
            Person-location values at which to draw reference lines.
        point_info_labels : bool, default False
            Annotate information line intersections.
        xmin, xmax : float
            Person-location axis limits.
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
        if isinstance(items, str) and items in ("all", "none"):
            items = None
        elif isinstance(items, str):
            items = [items]
        item_keys = list(self.responses.columns) if items is None else items
        estimates = np.arange(-20, 20, 0.1)
        y = np.array(
            [
                sum(
                    self.variance(a, self.items[it], self.thresholds)
                    for it in item_keys
                )
                for a in estimates
            ]
        ).reshape(-1, 1)
        if ymax is None:
            ymax = float(y.max()) * 1.1

        return self.plot_data(
            x_data=estimates,
            y_data=y,
            items=items,
            x_min=xmin,
            x_max=xmax,
            y_max=ymax,
            graph_title=title or "",
            point_info_lines_test=point_info_lines,
            score_labels=point_info_labels,
            y_label="Fisher information",
            plot_style=plot_style,
            palette=palette,
            black=black,
            font=font,
            title_font_size=title_font_size,
            axis_font_size=axis_font_size,
            labelsize=labelsize,
            filename=filename,
            plot_density=dpi,
            file_format=file_format,
        )

    def test_csem(
        self,
        items=None,
        point_csem_lines=None,
        point_csem_labels=False,
        xmin=-5,
        xmax=5,
        ymax=5,
        title=None,
        plot_style="white",
        palette="colorblind multi",
        black=False,
        font="Times New Roman",
        title_font_size=15,
        axis_font_size=12,
        labelsize=12,
        filename=None,
        file_format="png",
        dpi=300,
    ):
        """
        Plot the Test Conditional Standard Error of Measurement (CSEM) Curve.

        Displays 1 / sqrt(I(theta)) as a function of person location.

        Parameters
        ----------
        items : str, list, or None, default None
            Item subset. None uses all items.
        point_csem_lines : list or None, default None
            Person-location values at which to draw CSEM reference lines.
        point_csem_labels : bool, default False
            Annotate CSEM line intersections.
        xmin, xmax : float
            Person-location axis limits.
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
        if isinstance(items, str) and items in ("all", "none"):
            items = None
        elif isinstance(items, str):
            items = [items]
        item_keys = list(self.responses.columns) if items is None else items
        estimates = np.arange(-20, 20, 0.1)
        info = np.array(
            [
                sum(
                    self.variance(a, self.items[it], self.thresholds)
                    for it in item_keys
                )
                for a in estimates
            ]
        )
        y = (1.0 / (info**0.5)).reshape(-1, 1)

        return self.plot_data(
            x_data=estimates,
            y_data=y,
            items=items,
            x_min=xmin,
            x_max=xmax,
            y_max=ymax,
            graph_title=title or "",
            point_csem_lines=point_csem_lines,
            score_labels=point_csem_labels,
            y_label="Conditional SEM",
            plot_style=plot_style,
            palette=palette,
            black=black,
            font=font,
            title_font_size=title_font_size,
            axis_font_size=axis_font_size,
            labelsize=labelsize,
            filename=filename,
            plot_density=dpi,
            file_format=file_format,
        )

    def std_residuals_plot(
        self,
        items=None,
        bin_width=0.5,
        x_min=-6,
        x_max=6,
        normal=False,
        title=None,
        plot_style="white",
        black=False,
        font="Times New Roman",
        title_font_size=15,
        axis_font_size=12,
        labelsize=12,
        filename=None,
        file_format="png",
        plot_density=300,
    ):
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
        black : bool, default False
            If True, renders the histogram in black.
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
        if isinstance(items, str) and items in ("all", "none"):
            items = None
        elif isinstance(items, str):
            items = [items]

        std_residual_df = (
            self.std_residual_df if items is None else self.std_residual_df[items]
        )
        std_residual_list = std_residual_df.unstack().dropna()

        return self.std_residuals_hist(
            std_residual_list,
            bin_width=bin_width,
            x_min=x_min,
            x_max=x_max,
            normal=normal,
            title=title,
            plot_style=plot_style,
            black=black,
            font=font,
            title_font_size=title_font_size,
            axis_font_size=axis_font_size,
            labelsize=labelsize,
            filename=filename,
            file_format=file_format,
            plot_density=plot_density,
        )

    def wright_map(
        self,
        person_names=None,
        item_names=None,
        item_level="items",
        orientation="vertical",
        map_type="hist",
        item_labels=False,
        item_strip=False,
        item_distribution=False,
        strip_thickness=2.0,
        strip_alpha=0.3,
        sort="location",
        palette="colorblind multi",
        neutral_extremes=False,
        item_row_height=None,
        narrow_font=False,
        edge_padding=0.5,
        distribution_markers=False,
        group_by=None,
        group_colors=None,
        stack=True,
        blend=False,
        prop=False,
        person_lim=None,
        item_lim=None,
        person_scaling=1,
        no_of_bins=20,
        pad=False,
        plot_range=None,
        kde_points=500,
        bw_method="scott",
        line_width=1,
        figsize=None,
        title=None,
        plot_style="white",
        edge_color="black",
        person_color="skyblue",
        item_color="salmon",
        person_line_color="black",
        item_line_color="black",
        marker_color="darkred",
        alpha=0.6,
        black=False,
        font="Times New Roman",
        title_font_size=15,
        axis_font_size=12,
        labelsize=12,
        item_label_size=8,
        tick_interval=1,
        filename=None,
        file_format="png",
        dpi=300,
    ):
        """
        Plot a Wright map showing person and item location distributions.

        Displays the distribution of person locations alongside item
        difficulties on the same logit scale. Items can either mirror the
        person distribution on the opposite side of a zero baseline (as
        back-to-back histograms or KDE curves), or be listed by name in a
        dedicated panel, Winsteps-style, aligned to the same location axis.

        Parameters
        ----------
        person_names : str, list, or None, default None
            Person subset to include. None uses all persons.
        item_names : str, list, or None, default None
            Item subset to include. None uses all items.
        item_level : str, default 'items'
            'items' plots one point per item at its central location.
            'thresholds' plots one point per Rasch-Andrich threshold
            instead (RSM: delta_i + tau_k; PCM: thresholds_uncentred),
            flattened across all items. Ignored when item_strip=True,
            which always needs each item's full threshold set regardless
            of this setting.
        orientation : str, default 'vertical'
            'vertical' places location on the x-axis, persons above the
            baseline. 'horizontal' places location on the y-axis, persons
            to the left of the baseline.
        map_type : str, default 'hist'
            'hist' plots (back-to-back, when item_labels=False) histograms.
            'kde' plots smoothed kernel density estimate curves instead.
            Only governs the person side once item_labels=True, since the
            item side becomes a label panel rather than a distribution.
        item_labels : bool, default False
            If True, items are listed by name in a dedicated panel next to
            the person distribution (grouped into no_of_bins location bins,
            stacked as rows within a bin, ordered by location) rather than
            mirrored as a second distribution sharing the same axis.
        item_strip : bool, default False
            If True, draws each item as a Winsteps-style divided strip --
            one row per item, spanning its own threshold range, divided
            into coloured category segments with a boundary line at each
            threshold. Implies item_labels=True.
        item_distribution : bool, default False
            If True, adds a third panel -- a plain (non-mirrored) hist or
            KDE of item (or threshold, per item_level) locations -- between
            the person panel and the item_labels panel. Implies
            item_labels=True.
        strip_thickness : float, default 2.0
            Multiplier on the physical height (vertical orientation) or
            width (horizontal orientation) of each item's strip row. Only
            used when item_strip=True.
        strip_alpha : float, default 0.3
            Fill transparency for the strip's category segments. Only used
            when item_strip=True.
        sort : str, default 'location'
            Ordering of item strip rows. 'location' sorts by each item's
            lowest threshold. 'order' preserves self.items' own index
            order. Only used when item_strip=True.
        palette : str or None, default 'dark multi'
            Named colour palette for the strip's category segments,
            reusing the same palette_dict convention as plot_data(). None
            uses a flat item_color fill with no per-category distinction.
            Only used when item_strip=True.
        neutral_extremes : bool, default False
            The strip's open (unbounded) top and bottom categories always
            draw out to the location axis limits (there's no real boundary
            to stop short at). If True, they fade from a neutral cream
            colour toward the real threshold instead of using a flat fill
            in the category palette -- de-emphasising the open-ended
            extremes rather than drawing the viewer's eye to an arbitrary
            width. Only used when item_strip=True.
        item_row_height : float or None, default None
            Physical height, in inches, allocated per stacked item-label
            row. If None, measured automatically from the longest item
            name at the given font size. Only used when item_labels=True.
        narrow_font : bool, default False
            If True, item labels (and, with item_strip=True, the row
            labels beside the strip) are set in 'Arial Narrow' instead of
            font. A condensed font renders each name shorter, so it needs
            less stacked row height -- useful when item names are long
            and many end up sharing a bin or crowding the strip. Falls
            back to matplotlib's usual font substitution if 'Arial
            Narrow' isn't installed. Only affects item_labels=True;
            everything else (axis text, legend, title) still uses font.
        edge_padding : float, default 0.5
            Extra padding, in inches, added around the item-label margin
            (left of the axes in vertical orientation, below the axes in
            horizontal orientation) beyond what's needed to fit the
            longest item name. The same value produces matching-looking
            margins in both orientations. Only used when item_labels=True.
        distribution_markers : bool, default False
            If True, overlays mean/+-1SD/+-2SD reference marks (labelled
            mu, mu+-sigma, mu+-2sigma) for persons and items, in their
            respective colours.
        group_by : pandas.Series or None, default None
            Person-level grouping (e.g. a DIF group), indexed like
            self.persons, used to split the person distribution into one
            sub-distribution per unique value. None plots persons as a
            single series.
        group_colors : dict or None, default None
            Mapping from each group_by value to a colour. If None, colours
            are drawn from the 'tab10' colormap in sorted group order.
        stack : bool, default True
            When group_by is set: for map_type='hist', True draws one
            segmented column per bin (each group's count stacked within
            the bar); False draws separate columns per group (dodged, or
            alpha-blended if blend=True). For map_type='kde', True draws
            curves cumulatively stacked (streamgraph-style); False overlays
            each group's curve from zero.
        blend : bool, default False
            Only relevant when group_by is set, map_type='hist', and
            stack=False. If True, draws each group as full-height,
            alpha-blended overlapping bars instead of dodged (side-by-side)
            bars.
        prop : bool, default False
            If True, normalises each distribution to proportions/density
            rather than raw counts.
        person_lim : float or None, default None
            One-sided magnitude for persons' own Count/Density axis (e.g.
            person_lim=30 shows 0 to 30). If None, chosen automatically
            from persons' own natural peak, independently of items' own
            scale. Persons' own panel always keeps its full physical
            size regardless of this value -- items' own panel (when
            item_distribution=True) is sized proportionally against it,
            so a given Count/Density value reads at the same physical
            scale in both.
        item_lim : float or None, default None
            One-sided magnitude for items' own Count/Density axis. Same
            semantics as person_lim, for whichever panel items end up in
            (mirrored onto the same axis as persons when
            item_labels=False, or their own item_distribution panel).
        person_scaling : float, default 1
            Rebalances the item distribution against the person
            distribution: with person_scaling=10 the item side reads 10x
            larger per Count/Density unit than the person side, so the
            item distribution takes 10x more of the plot. Useful when far
            more persons than items otherwise leave the item distribution
            a flat, uninformative squiggle at the persons' scale. Both
            sides keep their true tick-label values either way. Applies
            wherever both distributions are drawn as hist/KDE. In the
            mirrored single-axis map (item_labels=False) the zero
            baseline simply shifts to give the item side more of the
            shared axis, at the same figure size -- the person side is
            compressed to make room. With a dedicated item_distribution
            panel (item_distribution=True) that panel is instead grown to
            match, enlarging the figure. No effect when items are shown
            only as labels (item_labels=True, item_distribution=False),
            where there is no item distribution to rebalance against.
        no_of_bins : int, default 20
            Number of histogram bins spanning the location range. Also
            defines the location-binning grid used for item_labels=True.
        pad : bool, default False
            If True, adds 5% padding to either end of the location axis
            limits. If False, the location axis is bounded exactly to the
            plotted range.
        plot_range : tuple of (float, float) or None, default None
            (lo, hi) limits for the location axis. If None, uses the floor
            of the combined minimum and the ceiling of the combined maximum
            of persons and items.
        kde_points : int, default 500
            Number of points at which each KDE curve is evaluated. Only
            used when map_type='kde'.
        bw_method : str, scalar, or callable, default 'scott'
            Bandwidth selection method passed to scipy.stats.gaussian_kde.
            Only used when map_type='kde'.
        line_width : float, default 1
            Line width of the KDE curves. Only used when map_type='kde'.
        figsize : tuple of (float, float) or None, default None
            Base figure size in inches for the person distribution. If
            None, defaults to (8, 6) for orientation='vertical' or (4, 8)
            for orientation='horizontal'. When item_labels=True, the person
            panel gets half of this (matching how much space it occupied
            in the original mirrored layout), and the figure grows further
            to fit however many stacked item rows are needed -- the person
            distribution's own size is never reduced to make room.
        title : str or None, default None
            Plot title. If None, no title is shown.
        plot_style : str, default 'white'
            Plot background style: 'white' or 'dark'.
        edge_color : str, default 'black'
            Edge colour of the histogram bars. Only used when
            map_type='hist' and group_by is None.
        person_color : str, default 'skyblue'
            Fill colour for the person distribution. Ignored when group_by
            is set (each group uses group_colors instead). Overridden by a
            grey shade when black=True.
        item_color : str, default 'salmon'
            Fill colour for the item distribution. Only used when
            item_labels=False. Overridden by a grey shade when black=True.
        person_line_color : str, default 'black'
            Line colour for the person KDE curve. Only used when
            map_type='kde' and group_by is None.
        item_line_color : str, default 'black'
            Colour for item labels (item_labels=True) or the item KDE
            curve line (item_labels=False, map_type='kde').
        marker_color : str, default 'darkred'
            Colour of the tick marks and labels drawn when
            distribution_markers=True. Only used for the single-series
            (group_by=None) case -- grouped marks use each group's own
            colour instead, to stay identifiable against its distribution.
        alpha : float, default 0.6
            Fill transparency for the distributions.
        black : bool, default False
            If True, renders person_color/item_color as grey shades
            instead. Has no effect on group_by colours.
        font : str, default 'Times New Roman'
            Font family for all plot text.
        title_font_size : int, default 15
            Title font size in points.
        axis_font_size : int, default 12
            Axis label font size in points.
        labelsize : int, default 12
            Tick label font size in points.
        item_label_size : int, default 8
            Font size, in points, for item names in the item_labels
            panel.
        tick_interval : float, default 1
            Spacing between Location axis ticks. Ticks are evenly spaced
            at multiples of this value across the plotted range, rather
            than using matplotlib's own automatic tick choice (which can
            land on an irregular spacing once the exact data min/max are
            also forced in as ticks).
        filename : str or None, default None
            If provided, saves the plot to this path. No file extension
            needed.
        file_format : str, default 'png'
            Output file format.
        dpi : int, default 300
            Output resolution in dots per inch.

        Returns
        -------
        None
            Displays and closes the figure. Use filename to save.
        """

        if black:
            person_color = "lightgray"
            item_color = "darkgray"

        if not hasattr(self, "persons"):
            self.person_estimates()

        persons = (
            self.persons if person_names is None else self.persons.loc[person_names]
        )

        # a strip needs every item's full threshold set regardless of
        # item_level (it draws the whole operating range, not one point),
        # so it also implies item_labels -- friendlier than raising when
        # someone passes item_strip=True on its own
        if item_strip:
            item_labels = True

        # a plain item/threshold hist or KDE panel only makes sense as a
        # third panel alongside the labelled one -- on its own it's just
        # the old mirrored mode with item_labels=False
        if item_distribution:
            item_labels = True

        base_items = self.items if item_names is None else self.items.loc[item_names]

        # the mirrored / item_distribution hist/KDE draws Rasch-Andrich
        # thresholds rather than item central locations whenever item_level
        # asks for them (or item_strip forces the per-threshold
        # flattening) -- label its legend entry to match.
        item_dist_label = (
            "Thresholds" if (item_level == "thresholds" or item_strip) else "Items"
        )

        if item_level == "items" and not item_strip:
            items = base_items
        else:
            # one entry per Rasch-Andrich threshold rather than one per item.
            # RSM shares a single threshold set (self.thresholds is a Series
            # of step values tau_k): delta_i + tau_k as one outer-sum matrix.
            # PCM has its own threshold set per item, already on the shared
            # absolute scale -- but as thresholds_uncentred, not
            # self.thresholds (which is centred *within* each item, so its
            # row means are ~0 and aren't comparable to self.items/persons
            # at all). Either way the result is a wide (item x threshold
            # number) frame.
            if isinstance(self.thresholds, pd.Series):
                matrix = pd.DataFrame(
                    base_items.values[:, None] + self.thresholds.values[None, :],
                    index=base_items.index,
                    columns=self.thresholds.index,
                )
            else:
                matrix = self.thresholds_uncentred.loc[base_items.index]

            # pandas' stack() stopped dropping NaN by default as of the
            # 2.1+ implementation, so drop the short items' padding NaNs
            # explicitly rather than relying on stack()'s own default
            stacked = matrix.stack().dropna()
            items = stacked.set_axis(
                [f"{item} (τ{k})" for item, k in stacked.index]
            )

            if item_strip:
                # kept grouped by item (not flattened) for the strip
                # renderer, which needs each item's own ordered threshold
                # list rather than one independent point per threshold.
                # Segments are the item's own *most probable category*
                # map -- the theta interval where each category actually
                # has the highest response probability -- rather than
                # just the interval between two raw threshold values.
                # Category k's (unnormalised) log-response-odds relative
                # to category 0 is S_k(theta) = k*theta - T_k, a straight
                # line in theta (T_k = the cumulative sum of thresholds
                # 1..k in their own natural, possibly disordered, order);
                # "most probable category" is exactly whichever line is
                # highest, so the segments are the upper envelope of
                # these lines. For ordered thresholds this reduces to
                # exactly the interval between adjacent thresholds (each
                # category's own probability curve is a single unimodal
                # "bump" peaking between its neighbours' crossings); when
                # disordered, a category whose bump never rises above its
                # neighbours' is correctly dropped from the envelope
                # entirely, rather than drawn as a spurious sliver.
                def _most_probable_category_segments(thresholds_natural):
                    m = len(thresholds_natural)
                    T = np.concatenate([[0.0], np.cumsum(thresholds_natural)])
                    hull = []

                    def redundant(l1, l2, l3):
                        m1, b1 = l1
                        m2, b2 = l2
                        m3, b3 = l3
                        return (b3 - b1) * (m1 - m2) <= (b2 - b1) * (m1 - m3)

                    # slopes (category numbers 0..m) are already sorted
                    # ascending regardless of threshold order, which is
                    # what lets this single-pass stack (the sorted-slope
                    # convex-hull trick) work without re-sorting anything
                    for h in range(m + 1):
                        line = (h, -T[h])
                        while len(hull) >= 2 and redundant(hull[-2], hull[-1], line):
                            hull.pop()
                        hull.append(line)
                    segs = []
                    for i, (slope, intercept) in enumerate(hull):
                        seg_lo = (
                            None
                            if i == 0
                            else (intercept - hull[i - 1][1])
                            / (hull[i - 1][0] - slope)
                        )
                        seg_hi = (
                            None
                            if i == len(hull) - 1
                            else (hull[i + 1][1] - intercept)
                            / (slope - hull[i + 1][0])
                        )
                        segs.append((seg_lo, seg_hi, slope))
                    return segs

                item_thresholds_natural = {
                    item: row.dropna().values for item, row in matrix.iterrows()
                }
                item_segments = {
                    item: _most_probable_category_segments(t)
                    for item, t in item_thresholds_natural.items()
                }
                # a category missing from its item's segments never has
                # the highest response probability anywhere -- exactly
                # what disordered Rasch-Andrich thresholds mean
                # geometrically (its own probability "bump" never rises
                # above its neighbours')
                item_missing_categories = {
                    item: sorted(
                        set(range(len(t) + 1))
                        - {lab for _, _, lab in item_segments[item]}
                    )
                    for item, t in item_thresholds_natural.items()
                }
                disordered_items = [
                    item
                    for item, missing in item_missing_categories.items()
                    if missing
                ]
                if disordered_items:
                    warnings.warn(
                        "Disordered thresholds for item(s) "
                        f"{', '.join(str(i) for i in disordered_items)}. "
                        "Only categories that are most probable for some "
                        "range of the trait shown. For more detailed "
                        "inspection of threshold structure, run self.crcs() "
                        "plot and/or self.threshold_stats_df()."
                    )

        no_of_items = len(items)

        if group_by is not None:
            group_by = group_by.reindex(persons.index)
            person_group_values = sorted(group_by.dropna().unique(), key=str)
            if group_colors is None:
                cmap = plt.get_cmap("tab10")
                group_colors = {
                    g: cmap(i % 10) for i, g in enumerate(person_group_values)
                }

        if plot_range is None:
            combined = np.concatenate([persons.values, items.values])
            span_lo, span_hi = float(combined.min()), float(combined.max())
            if distribution_markers:
                # the μ±2σ ticks (and their rotated labels) are drawn at
                # mean ± 2·SD of each plotted distribution and can fall
                # outside its raw data extent when that distribution is
                # wide or heavy-tailed — fold them into the range so they
                # stay on-canvas rather than being clipped at the spine
                marker_values = [items.values]
                if group_by is not None:
                    marker_values += [
                        persons.values[group_by.values == g]
                        for g in person_group_values
                    ]
                else:
                    marker_values.append(persons.values)
                for mv in marker_values:
                    if len(mv):
                        mv_mean, mv_sd = float(np.mean(mv)), float(np.std(mv))
                        span_lo = min(span_lo, mv_mean - 2 * mv_sd)
                        span_hi = max(span_hi, mv_mean + 2 * mv_sd)
            lo = np.floor(span_lo)
            hi = np.ceil(span_hi)
        else:
            lo, hi = plot_range[0], plot_range[1]

        bins = np.linspace(lo, hi, no_of_bins + 1)

        # each distribution's own natural peak, computed analytically (no
        # plotting needed) so panel sizing/axis ranges can be resolved
        # before the figure is even built. Computed independently per
        # distribution -- deliberately NOT sharing a single "widest"
        # value across persons/items, since a small item set with a very
        # different natural scale would otherwise force persons' own
        # well-populated distribution onto a needlessly inflated (or
        # cramped) scale
        def _natural_max(values, n):
            if map_type == "hist":
                counts, _ = np.histogram(values, bins=bins)
                peak = counts.max() if len(counts) else 0
                return peak / n if prop else peak
            kde_vals = gaussian_kde(values, bw_method=bw_method)(
                np.linspace(lo, hi, kde_points)
            )
            peak = kde_vals.max() if len(kde_vals) else 0
            return peak if prop else peak * n

        person_natural_max = _natural_max(persons.values, len(persons))
        item_natural_max = _natural_max(items.values, no_of_items)

        # rounds a natural peak up to a "neat" ceiling -- 1/1.5/2/2.5/3/
        # 4/5/6/8/10 x a power of 10 -- and picks a tick step that
        # divides it exactly, from a matching per-fraction divisor,
        # rather than leaving the step to a separately-chosen locator
        # that has no reason to land on a divisor of this specific
        # ceiling (which previously left the axis's own true top edge
        # short of its last drawn tick, e.g. ticks 0..28 against a
        # ceiling of 30). The 10% pad before searching guarantees the
        # chosen ceiling sits visibly above the true peak even when that
        # peak already IS a neat number itself (a small integer count,
        # for instance) -- without it, the tallest bar/curve would land
        # flush against the axis's own edge, with no headroom at all.
        _nice_fractions = [1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10]
        _nice_divisors = {1: 4, 1.5: 3, 2: 4, 2.5: 5, 3: 3, 4: 4, 5: 5, 6: 3, 8: 4, 10: 5}

        def _nice_ceil_and_step(x, integer):
            if x <= 0:
                return 0, (1 if integer else 0.1)
            x = x * 1.1
            exponent = np.floor(np.log10(x))
            fraction = x / 10**exponent
            nice_fraction = next(f for f in _nice_fractions if f >= fraction - 1e-9)
            ceiling = nice_fraction * 10**exponent
            step = ceiling / _nice_divisors[nice_fraction]
            if integer:
                step = max(1, round(step))
                ceiling = step * np.ceil(x / step)
            return ceiling, step

        def _nice_step_for(ceiling, integer):
            # an explicit *_lim is used exactly as given (the ceiling
            # itself isn't adjusted), but still gets a step matched to
            # whichever known "nice" fraction it's closest to, so it at
            # least *usually* divides evenly -- an arbitrary user value
            # isn't guaranteed to, the way an auto-computed ceiling is
            if ceiling <= 0:
                return 1 if integer else 0.1
            exponent = np.floor(np.log10(ceiling))
            fraction = ceiling / 10**exponent
            nice_fraction = min(_nice_fractions, key=lambda f: abs(f - fraction))
            step = ceiling / _nice_divisors[nice_fraction]
            if integer:
                step = max(1, round(step))
            return step

        _integer_counts = map_type == "hist" and not prop

        if person_lim is not None:
            person_range = person_lim
            person_step = _nice_step_for(person_lim, _integer_counts)
        else:
            person_range, person_step = _nice_ceil_and_step(
                person_natural_max, _integer_counts
            )
        if item_lim is not None:
            item_range = item_lim
            item_step = _nice_step_for(item_lim, _integer_counts)
        else:
            item_range, item_step = _nice_ceil_and_step(item_natural_max, _integer_counts)

        # In the mirrored single-axis map (item_labels=False) persons and
        # items share one count axis, so person_scaling has no separate
        # panel to resize the way it does with item_distribution=True.
        # Instead it enlarges the item side of that shared axis by this
        # factor: the item distribution ends up person_scaling x larger
        # relative to persons, with the figure size unchanged (the person
        # side is compressed to make room). 1 elsewhere, so the
        # item_labels / item_distribution paths are untouched.
        mirror_item_scale = person_scaling if not item_labels else 1

        # breathing room reserved before row 0 of the item panel (used for
        # both the boundary_margin below and, when distribution_markers
        # draws a tick alongside each item-side mark, for keeping that
        # tick inside the margin and clear of row 0's own text -- shared
        # so the two stay consistent if this value ever changes)
        item_row_margin = 0.45

        # shared physical sizing for distribution_markers' ticks: a fixed
        # gap (inches) between a tick and its own label, and a fixed tick
        # length (rows) on the item side -- both fixed in real inches
        # rather than a fraction of whatever data range is being plotted,
        # so the two sides (persons' count-axis-based scale vs items'
        # row-based scale) end up visually consistent instead of drifting
        # apart depending on the dataset
        mark_gap_in = 0.065
        mark_tick_rows = 0.15
        # physical breathing room (inches) between the widest mark's own
        # label and where item names start -- separate from the +0.5-row
        # collision buffer below (that one exists purely to stop an
        # item's own centred text reaching back into the mark's text; this
        # one is just visual spacing between the two blocks of text)
        item_label_gap_in = 0.08

        # when item_labels shows a dedicated item panel (rather than
        # item_distribution's separate histogram/KDE), distribution_markers'
        # item-side marks are drawn straight into that panel's own row
        # grid, all at row 0 -- items then start uniformly at row 1,
        # regardless of which bin(s) the marks themselves land in. Every
        # mark sitting at the same row is what keeps their own starting
        # point aligned with each other (not just with the items); if two
        # marks happened to share a bin and got stacked into different
        # rows instead, their own gap to the boundary would no longer
        # match, even though items would still be fine. Only meaningful
        # for the binned (non-strip) item panel -- item_strip rows are
        # one-per-item already, not shared with anything.
        item_mark_rows = {}
        mark_row_depth = 0
        if distribution_markers and item_labels and not item_distribution and not item_strip:
            item_mean = items.mean()
            item_sd = items.std()
            for mark_label, mark_loc in (
                ("μ", item_mean),
                ("μ−σ", item_mean - item_sd),
                ("μ+σ", item_mean + item_sd),
                ("μ−2σ", item_mean - 2 * item_sd),
                ("μ+2σ", item_mean + 2 * item_sd),
            ):
                item_mark_rows[mark_label] = mark_loc
            mark_row_depth = 1

        if figsize is None:
            base_w, base_h = (8, 6) if orientation == "vertical" else (4, 8)
        else:
            base_w, base_h = figsize

        style_overrides = {"font.family": font, "font.size": axis_font_size}

        with plt.rc_context(style_overrides):

            if plot_style != "white":
                self._apply_plot_style(plot_style)

            label_margin_in = 0.0
            item_font = "Arial Narrow" if narrow_font else font

            if item_labels and item_strip:
                # divided-strip layout: each item is one row spanning its
                # own threshold range. Row labels are drawn once per row,
                # like a tick label, so each item always gets its own row
                # (no packing) to keep that labelling unambiguous.
                strip_ext = (hi - lo) * 0.03

                def _draw_bounds(segs):
                    # extend the open top/bottom categories by the width
                    # of the adjacent *bounded* segment (falling back to
                    # the small strip_ext buffer when there's only the
                    # pair of extreme segments to work with -- either a
                    # genuinely single-threshold item, or one where every
                    # intermediate category has been dropped as
                    # disordered, leaving nothing to borrow a width from)
                    if len(segs) >= 3:
                        return (
                            segs[0][1] - (segs[1][1] - segs[1][0]),
                            segs[-1][0] + (segs[-2][1] - segs[-2][0]),
                        )
                    return segs[0][1] - strip_ext, segs[-1][0] + strip_ext

                item_spans = {
                    name: _draw_bounds(segs) for name, segs in item_segments.items()
                }

                if sort == "location":
                    ordered_names = sorted(item_spans, key=lambda name: item_spans[name][0])
                else:
                    ordered_names = list(item_thresholds_natural.keys())

                item_rows = {name: i for i, name in enumerate(ordered_names)}
                max_depth = len(ordered_names)

                # names are labelled like axis ticks -- one label per row,
                # at a fixed position outside the plot area, rather than
                # inline after each item's own (variably-positioned) bar.
                row_names = {}
                for name in ordered_names:
                    row_names.setdefault(item_rows[name], []).append(name)
                row_labels = {row: ", ".join(names) for row, names in row_names.items()}

                widest_label = max(row_labels.values(), key=len)
                probe_fig = plt.figure()
                probe_ax = probe_fig.add_subplot(111)
                label_rotation = 0 if orientation == "vertical" else 90
                probe_text = probe_ax.text(
                    0,
                    0,
                    widest_label,
                    fontsize=labelsize,
                    fontfamily=item_font,
                    rotation=label_rotation,
                    rotation_mode="anchor",
                )
                probe_fig.canvas.draw()
                bbox = probe_text.get_window_extent(
                    renderer=probe_fig.canvas.get_renderer()
                )
                label_dim = bbox.width if orientation == "vertical" else bbox.height
                label_margin_in = label_dim / probe_fig.dpi + edge_padding
                plt.close(probe_fig)

                if item_row_height is None:
                    # strip rows use horizontal (unrotated) text, so row
                    # height only depends on font line-height, not on how
                    # long any given item name is
                    probe_fig = plt.figure()
                    probe_ax = probe_fig.add_subplot(111)
                    probe_text = probe_ax.text(
                        0, 0, "Ag", fontsize=labelsize, fontfamily=item_font
                    )
                    probe_fig.canvas.draw()
                    bbox = probe_text.get_window_extent(
                        renderer=probe_fig.canvas.get_renderer()
                    )
                    # row spacing/bar_half stay fixed in data-units; giving
                    # each row more physical inches (via strip_thickness)
                    # is what actually makes the bar thicker on screen --
                    # scaling bar_half instead would just repartition the
                    # same physical row into a bigger bar/smaller gap, with
                    # no net change to how thick it looks
                    row_height_in = bbox.height / probe_fig.dpi * 1.6 * strip_thickness
                    plt.close(probe_fig)
                else:
                    row_height_in = item_row_height * strip_thickness

                # the axis itself later spans (max_depth + 0.1 +
                # boundary_margin) row-units, not just max_depth -- sizing
                # the panel to only max_depth*row_height_in would leave
                # every row rendered slightly smaller than row_height_in
                # actually intended, quietly eating into the buffer built
                # into row_height_in's own measurement
                item_panel_in = (
                    max_depth + 0.1 + (0.7 if distribution_markers else 0.45)
                ) * row_height_in

            elif item_labels:
                # Winsteps-style: bin items onto the histogram grid and
                # stack names as rows within each occupied bin. Figuring
                # out how many rows are needed happens up front, before any
                # figure is created, so the item panel can be sized in real
                # inches rather than measured and solved for after the fact
                # -- no circular dependency, and it scales to any number of
                # items.
                bin_idx = np.clip(
                    np.digitize(items.values, bins) - 1, 0, no_of_bins - 1
                )
                bin_centers = (bins[:-1] + bins[1:]) / 2
                item_groups = {}
                for (name, loc), b in zip(items.items(), bin_idx):
                    item_groups.setdefault(b, []).append((loc, name))

                if item_row_height is None:
                    # a rotated string's rendered length depends on how
                    # many characters it has -- measure the longest name
                    # once, on a throwaway probe figure using the same font
                    # context, entirely separate from the real figure, so
                    # there's no risk of the measurement going stale when
                    # the real figure's size is set afterward
                    longest_name = max((str(n) for n in items.index), key=len)
                    probe_fig = plt.figure()
                    probe_ax = probe_fig.add_subplot(111)
                    probe_text = probe_ax.text(
                        0,
                        0,
                        longest_name,
                        rotation=90,
                        fontsize=item_label_size,
                        fontfamily=item_font,
                        rotation_mode="anchor",
                    )
                    probe_fig.canvas.draw()
                    bbox = probe_text.get_window_extent(
                        renderer=probe_fig.canvas.get_renderer()
                    )
                    row_height_in = bbox.height / probe_fig.dpi * 1.3
                    plt.close(probe_fig)
                else:
                    row_height_in = item_row_height

                if item_mark_rows:
                    # marks are now left/top-anchored at row 0 (see the
                    # distribution_markers block below) rather than
                    # centred, so the widest one ("mu+2sigma") reaches a
                    # full row_height_in-equivalent into row-space, not
                    # just half of it -- mark_row_depth has to cover that
                    # full reach, rounded up to a whole row, or a long
                    # mark's own label can run into row 1's item text.
                    # Item names are themselves centred on their own row
                    # (below), reaching back up to half a row from their
                    # centre, so padding by that same half-row is what
                    # actually keeps the two apart, not just the mark's
                    # own raw length.
                    probe_fig = plt.figure()
                    probe_ax = probe_fig.add_subplot(111)
                    probe_text = probe_ax.text(
                        0, 0, "μ+2σ", rotation=90, fontsize=labelsize,
                        rotation_mode="anchor",
                    )
                    probe_fig.canvas.draw()
                    bbox = probe_text.get_window_extent(
                        renderer=probe_fig.canvas.get_renderer()
                    )
                    mark_label_in = bbox.height / probe_fig.dpi
                    plt.close(probe_fig)
                    # marks' own text starts right after their tick (which
                    # sits flush with the panel's outer edge at
                    # -item_row_margin, plus its own length and a fixed
                    # physical gap converted to rows) -- not at row 0;
                    # row 0 is only where item names start stacking. See
                    # mark_gap_in/mark_tick_rows above and the drawing
                    # code below, which both need this same value. Left
                    # as a float (not rounded up to a whole row): items
                    # stack in whole-row steps relative to *each other*
                    # via i + mark_row_depth, which works just as well
                    # from a fractional starting depth, and rounding up
                    # here only wastes up to a full row of empty space
                    # that was never actually needed
                    mark_gap_rows = mark_gap_in / row_height_in
                    mark_text_row = -item_row_margin + mark_tick_rows + mark_gap_rows
                    # item_label_gap_in is purely visual spacing on top of
                    # the collision buffer -- without it, the buffer alone
                    # can leave the mark's text and the item names
                    # touching with no breathing room
                    mark_row_depth = max(
                        1.0,
                        mark_text_row
                        + mark_label_in / row_height_in
                        + item_label_gap_in / row_height_in
                        + 0.5,
                    )
                else:
                    # exact same formula as mark_text_row above with the
                    # mark-specific terms (mark_tick_rows, mark_gap_rows,
                    # the mark's own text reach) zeroed out, since there's
                    # no mark here to reach past -- the earlier version of
                    # this branch dropped the leading -item_row_margin
                    # entirely, which mark_text_row always includes, so it
                    # was measuring the gap from row 0 rather than from
                    # the actual boundary and came out several times too
                    # big. Same +0.5 collision buffer as above, for the
                    # same reason (an item's own centred text reaching
                    # back half a row from its centre)
                    mark_row_depth = (
                        -item_row_margin + item_label_gap_in / row_height_in + 0.5
                    )

                # every bin's items start after the same mark_row_depth
                # band (see above), so the deepest bin sets the panel's
                # depth regardless of which specific bin(s) hold marks
                max_depth = max(
                    max((len(v) for v in item_groups.values()), default=0)
                    + mark_row_depth,
                    1,
                )
                # the axis itself later spans (max_depth + 0.1 +
                # item_row_margin) row-units, not just max_depth -- sizing
                # the panel to only max_depth*row_height_in would leave
                # every row rendered slightly smaller than row_height_in
                # actually intended, quietly eating into the buffer built
                # into row_height_in's own measurement (most visible on
                # whichever item lands in the last row, right against the
                # panel's outer edge)
                item_panel_in = (max_depth + 0.1 + item_row_margin) * row_height_in
            else:
                item_panel_in = 0

            # figsize originally described the whole mirrored plot (persons
            # above, items below, sharing base_h/base_w). Now that items
            # get their own panel, persons only need the half of that
            # budget they used to occupy -- and persons' own panel always
            # gets that full share regardless of person_range, per the
            # docstring's promise that it never shrinks to make room.
            # item_distribution's panel (always alongside a one-sided ax,
            # since item_distribution implies item_labels=True) is sized
            # proportionally to item_range vs person_range, so a given
            # Count/Density value ends up at the same physical scale in
            # both -- e.g. an item panel needing half of persons' own
            # range gets half of persons' own physical size, not a fixed
            # fraction of the whole figure regardless of what it needs.
            person_panel_h = base_h / 2 if item_labels else base_h
            person_panel_w = base_w / 2 if item_labels else base_w
            item_dist_panel_h = (
                person_panel_h * (item_range / person_range) * person_scaling
                if item_distribution and person_range > 0
                else (person_panel_h if item_distribution else 0)
            )
            item_dist_panel_w = (
                person_panel_w * (item_range / person_range) * person_scaling
                if item_distribution and person_range > 0
                else (person_panel_w if item_distribution else 0)
            )

            # The persons panel always gets its full intended size -- adding
            # the item panel(s) grows the figure, it never shrinks the
            # distribution's own space. height_ratios/width_ratios alone
            # only fix the *ratio* between panels within whatever plottable
            # area matplotlib's default margins leave -- they don't pin
            # absolute inches, so pinning top/bottom (or left/right)
            # margins in inches is needed to make the split exact.
            if not item_labels:
                fig, ax = plt.subplots(figsize=(base_w, base_h))
                ax_items = None
                ax_item_dist = None
            elif orientation == "vertical":
                # extra top margin makes room for the duplicate Location
                # tick row now drawn on ax's own top edge, alongside the
                # title, plus a bit more breathing room below the title
                # itself. Sized (empirically, against the horizontal
                # branch's own margin below) so the title-to-content gap
                # matches horizontal's despite the tick row eating into
                # this orientation's margin budget that horizontal doesn't
                # have to spend.
                margin_top_in, margin_bottom_in = 1.0, 0.6
                # right edge gets the same plain edge_padding as any other
                # side with nothing to fit -- without an explicit value
                # here, matplotlib's own default leftover-space margin
                # takes over and is usually much bigger than the other
                # three sides
                margin_right_in = edge_padding
                fig_w = base_w + label_margin_in + margin_right_in
                fig_h = (
                    person_panel_h
                    + item_dist_panel_h
                    + item_panel_in
                    + margin_top_in
                    + margin_bottom_in
                )
                fig = plt.figure(figsize=(fig_w, fig_h))
                if item_distribution:
                    nrows = 3
                    height_ratios = [person_panel_h, item_dist_panel_h, item_panel_in]
                else:
                    nrows = 2
                    height_ratios = [person_panel_h, item_panel_in]
                gridspec_kwargs = dict(
                    height_ratios=height_ratios,
                    hspace=0,
                    top=1 - margin_top_in / fig_h,
                    bottom=margin_bottom_in / fig_h,
                    right=1 - margin_right_in / fig_w,
                )
                if label_margin_in:
                    # room for the strip's row labels, drawn just left of
                    # the axes like a column of y-tick labels
                    gridspec_kwargs["left"] = label_margin_in / fig_w
                gs = fig.add_gridspec(nrows, 1, **gridspec_kwargs)
                ax = fig.add_subplot(gs[0])
                if item_distribution:
                    ax_item_dist = fig.add_subplot(gs[1], sharex=ax)
                    ax_items = fig.add_subplot(gs[2], sharex=ax)
                else:
                    ax_item_dist = None
                    ax_items = fig.add_subplot(gs[1], sharex=ax)
            else:
                # right margin matches left: ax_items' Location scale (ticks
                # + title) now lives on its outer (right) edge, mirroring
                # ax's own Location scale on the left, so it needs the same
                # amount of room
                margin_left_in, margin_right_in = 0.7, 0.7
                # top margin was previously unreserved -- fine while the
                # title sat on ax's own axes, but now that it's a
                # figure-wide suptitle (see below, so it centres over the
                # whole figure rather than just the persons panel) it
                # needs real room. Sized for a 50% bigger title-to-plot
                # gap than the original unreserved default gave.
                margin_top_in = 0.67
                fig_w = (
                    person_panel_w
                    + item_dist_panel_w
                    + item_panel_in
                    + margin_left_in
                    + margin_right_in
                )
                fig_h = base_h + label_margin_in + margin_top_in
                fig = plt.figure(figsize=(fig_w, fig_h))
                if item_distribution:
                    ncols = 3
                    width_ratios = [person_panel_w, item_dist_panel_w, item_panel_in]
                else:
                    ncols = 2
                    width_ratios = [person_panel_w, item_panel_in]
                gridspec_kwargs = dict(
                    width_ratios=width_ratios,
                    wspace=0,
                    left=margin_left_in / fig_w,
                    right=1 - margin_right_in / fig_w,
                    top=1 - margin_top_in / fig_h,
                )
                if label_margin_in:
                    # room for the strip's row labels, drawn just below
                    # the axes like a row of x-tick labels. Same formula as
                    # vertical's left margin below -- both are driven
                    # purely by label_margin_in (text width/height plus
                    # edge_padding), so the two orientations' edge gaps
                    # match rather than horizontal getting an extra fixed
                    # 0.3in on top.
                    gridspec_kwargs["bottom"] = label_margin_in / fig_h
                gs = fig.add_gridspec(1, ncols, **gridspec_kwargs)
                ax = fig.add_subplot(gs[0])
                if item_distribution:
                    ax_item_dist = fig.add_subplot(gs[1], sharey=ax)
                    ax_items = fig.add_subplot(gs[2], sharey=ax)
                else:
                    ax_item_dist = None
                    ax_items = fig.add_subplot(gs[1], sharey=ax)

            # Persons always plot using the same back-to-back convention,
            # whether or not there's a dedicated item panel: zero-at-
            # boundary, bulk growing away from it. In horizontal
            # orientation the boundary with any item panel (item_labels or
            # item_distribution) sits on persons' right, so unflipped
            # (sign=+1) would put persons' zero at its own outer-left edge
            # instead -- the opposite of the back-to-back look. Flipping
            # (sign=-1) puts persons' zero at the boundary, matching item
            # panels and back-to-back histograms alike. Vertical orientation
            # never needs this: persons' natural zero-at-bottom already
            # sits at its boundary with any panel below.
            person_sign = 1 if orientation == "vertical" else -1

            # items draw on their own axis in two cases: the old mirrored
            # single-axes mode (item_labels=False), or the new
            # item_distribution panel -- both flip the sign the same way
            # (downward for vertical, rightward for horizontal) so the
            # item_distribution panel reads as the same "flipped" shape as
            # the old SLM-style mirrored map, just moved into its own
            # panel instead of sharing ax with persons. Otherwise
            # (item_labels=True, item_distribution=False) there's no item
            # hist/kde at all -- just the labelled panel below.
            item_sign = -1 if orientation == "vertical" else 1
            if not item_labels:
                item_ax = ax
            elif item_distribution:
                item_ax = ax_item_dist
            else:
                item_ax = None
                item_sign = None

            if map_type == "kde":
                x_grid = np.linspace(lo, hi, kde_points)

            if group_by is None:
                person_groups = [("Persons", persons, person_color, person_line_color)]
            else:
                person_groups = [
                    (str(g), persons[group_by == g], group_colors[g], group_colors[g])
                    for g in person_group_values
                ]

            if map_type == "hist":
                if stack or group_by is None:
                    # single series, or one segmented column per bin
                    data = [sub for _, sub, _, _ in person_groups]
                    weights = [
                        person_sign * np.ones_like(sub) / (len(sub) if prop else 1)
                        for sub in data
                    ]
                    bar_colors = [c for _, _, c, _ in person_groups]
                    group_labels = [label for label, _, _, _ in person_groups]

                    ax.hist(
                        data,
                        weights=weights,
                        bins=bins,
                        color=bar_colors,
                        edgecolor=edge_color,
                        label=group_labels,
                        alpha=alpha,
                        orientation=orientation,
                        stacked=True,
                    )

                elif not blend:
                    # dodged: matplotlib's native side-by-side bars for
                    # multiple datasets in one hist() call -- each group
                    # gets its own slice of the bin width, no colour
                    # blending to interpret
                    data = [sub for _, sub, _, _ in person_groups]
                    weights = [
                        person_sign * np.ones_like(sub) / (len(sub) if prop else 1)
                        for sub in data
                    ]
                    bar_colors = [c for _, _, c, _ in person_groups]
                    group_labels = [label for label, _, _, _ in person_groups]

                    ax.hist(
                        data,
                        weights=weights,
                        bins=bins,
                        color=bar_colors,
                        edgecolor=edge_color,
                        label=group_labels,
                        alpha=alpha,
                        orientation=orientation,
                        stacked=False,
                    )

                else:
                    # blend: each group's own full-height bars,
                    # alpha-blended on top of each other rather than
                    # dodged or stacked
                    for label, sub, fill_color, line_color in person_groups:
                        weights = person_sign * np.ones_like(sub)
                        if prop:
                            weights = weights / len(sub)

                        ax.hist(
                            sub,
                            weights=weights,
                            bins=bins,
                            color=fill_color,
                            edgecolor=edge_color,
                            label=label,
                            alpha=alpha,
                            orientation=orientation,
                        )

            else:
                cumulative = np.zeros_like(x_grid)
                for label, sub, fill_color, line_color in person_groups:
                    n_sub = len(sub)
                    kde_sub = gaussian_kde(sub, bw_method=bw_method)(x_grid)
                    curve = person_sign * (kde_sub if prop else kde_sub * n_sub)

                    if stack:
                        # streamgraph-style: each group's fill runs from the
                        # running total up to running total + its own
                        # curve, rather than every curve overlaying from
                        # zero
                        base, top = cumulative, cumulative + curve
                        cumulative = top
                    else:
                        base, top = np.zeros_like(x_grid), curve

                    if orientation == "vertical":
                        ax.plot(x_grid, top, color=line_color, linewidth=line_width)
                        ax.fill_between(
                            x_grid,
                            base,
                            top,
                            color=fill_color,
                            alpha=alpha,
                            edgecolor=line_color,
                            linewidth=line_width,
                            label=label,
                        )
                    else:
                        ax.plot(top, x_grid, color=line_color, linewidth=line_width)
                        ax.fill_betweenx(
                            x_grid,
                            base,
                            top,
                            color=fill_color,
                            alpha=alpha,
                            edgecolor=line_color,
                            linewidth=line_width,
                            label=label,
                        )

            # --- items: a hist/kde on their own axis (either the old
            # mirrored ax, sign-flipped, or the new item_distribution
            # panel, unflipped), or labelled rows in a dedicated panel, or
            # both at once ---
            if item_ax is not None:
                if map_type == "hist":
                    item_weights = item_sign * mirror_item_scale * np.ones_like(items)
                    if prop:
                        item_weights = item_weights / no_of_items

                    item_ax.hist(
                        items,
                        weights=item_weights,
                        bins=bins,
                        color=item_color,
                        edgecolor=edge_color,
                        label=item_dist_label,
                        alpha=alpha,
                        orientation=orientation,
                    )

                else:
                    kde_items = gaussian_kde(items, bw_method=bw_method)(x_grid)
                    item_curve = item_sign * mirror_item_scale * (
                        kde_items if prop else kde_items * no_of_items
                    )

                    if orientation == "vertical":
                        item_ax.plot(
                            x_grid,
                            item_curve,
                            color=item_line_color,
                            linewidth=line_width,
                        )
                        item_ax.fill_between(
                            x_grid,
                            0,
                            item_curve,
                            color=item_color,
                            alpha=alpha,
                            edgecolor=item_line_color,
                            linewidth=line_width,
                            label=item_dist_label,
                        )
                    else:
                        item_ax.plot(
                            item_curve,
                            x_grid,
                            color=item_line_color,
                            linewidth=line_width,
                        )
                        item_ax.fill_betweenx(
                            x_grid,
                            0,
                            item_curve,
                            color=item_color,
                            alpha=alpha,
                            edgecolor=item_line_color,
                            linewidth=line_width,
                            label=item_dist_label,
                        )

            if item_distribution:
                # item_dist sits between the persons panel above and the
                # labelled panel below -- drop its near-side spine (facing
                # ax) for the same reason ax_items drops its own further
                # down, and hide its shared location-axis tick labels since
                # the bottom-most panel (ax_items) is the one that shows
                # them
                if orientation == "vertical":
                    ax_item_dist.spines["top"].set_visible(False)
                    ax_item_dist.tick_params(labelbottom=False)
                else:
                    ax_item_dist.spines["left"].set_visible(False)
                    ax_item_dist.tick_params(labelleft=False)

            # the labelled panel is independent of whether item_ax drew a
            # hist/kde above -- item_distribution can add that panel on
            # top of either labelling style, so this is its own if/elif
            # rather than chained onto the item_ax branch above
            if item_labels and item_strip:
                # row 0 sits nearest the boundary with the persons panel,
                # deeper rows move further away. Widened when
                # distribution_markers=True so row 0 doesn't crowd the
                # item mu/sigma marks overhanging from the persons panel
                # just above.
                boundary_margin = 0.7 if distribution_markers else 0.45
                if orientation == "vertical":
                    ax_items.set_ylim(max_depth + 0.1, -boundary_margin)
                else:
                    ax_items.set_xlim(-boundary_margin, max_depth + 0.1)

                # palette=None: flat fill, no per-category distinction
                # beyond the divider lines. palette=<name>: reuse the exact
                # same named-palette -> colormap mechanism plot_data() uses
                # for category response curves elsewhere in RaschPy, so a
                # given palette name looks the same on this plot as on a
                # crcs()/icc() plot.
                if palette is not None:
                    palette_dict = {
                        "dark blue": ["dark", "royalblue"],
                        "light blue": ["light", "cornflowerblue"],
                        "dark red": ["dark", "firebrick"],
                        "light red": ["light", "indianred"],
                        "dark green": ["dark", "forestgreen"],
                        "light green": ["light", "mediumseagreen"],
                        "dark grey": ["dark", "dimgrey"],
                        "light grey": ["light", "darkgrey"],
                        "dark multi": ["dark", "dark"],
                        "light multi": ["light", "muted"],
                        "colorblind multi": ["dark", "colorblind"],
                    }
                    shade, base_color = palette_dict[palette]
                    if palette == "colorblind multi":
                        # Okabe & Ito (2008) -- colour-vision-deficiency-safe
                        # qualitative palette, the scientific-publishing standard.
                        # The 8th (black) entry is swapped for white on dark
                        # backgrounds, where it would otherwise be invisible.
                        color_map = [
                            "#E69F00", "#56B4E9", "#009E73", "#F0E442",
                            "#0072B2", "#D55E00", "#CC79A7",
                            "#FFFFFF" if plot_style in self._DARK_BACKGROUND_STYLES else "#000000",
                        ]
                    elif shade == "dark":
                        color_map = (
                            sns.color_palette("dark", as_cmap=True)
                            if palette == "dark multi"
                            else sns.dark_palette(base_color, reverse=True, as_cmap=True)
                        )
                    else:
                        color_map = (
                            sns.color_palette("muted", as_cmap=True)
                            if palette == "light multi"
                            else sns.light_palette(
                                base_color, reverse=True, as_cmap=True
                            )
                        )
                    max_categories = (
                        max(len(t) for t in item_thresholds_natural.values()) + 1
                    )
                    cNorm = colors.Normalize(vmin=0, vmax=max_categories + 2)
                    if "multi" not in palette:
                        scalar_map = cmx.ScalarMappable(norm=cNorm, cmap=color_map)

                    def category_color(k):
                        return (
                            scalar_map.to_rgba(k)
                            if "multi" not in palette
                            else color_map[k]
                        )

                else:

                    def category_color(k):
                        return item_color

                bar_half = 0.3
                min_label_width = (hi - lo) * 0.02

                for name, segs in item_segments.items():
                    row = item_rows[name]
                    # the open (unbounded) top/bottom categories always draw
                    # out to the panel edge -- there's no real boundary to
                    # stop at, so stopping partway (at item_spans' own
                    # adjacent-width bound) just left an unexplained gap of
                    # blank background between the strip and the axis edge.
                    # neutral_extremes only controls whether that edge fill
                    # is a flat block in the category colour (False) or
                    # fades to a neutral colour (True), not whether it
                    # reaches the edge at all.
                    edge_lo, edge_hi = lo, hi

                    n_segs = len(segs)
                    for i, (seg_lo, seg_hi, k) in enumerate(segs):
                        x0 = edge_lo if seg_lo is None else seg_lo
                        x1 = edge_hi if seg_hi is None else seg_hi
                        is_extreme = neutral_extremes and (i == 0 or i == n_segs - 1)

                        if is_extreme:
                            # fade a neutral colour in from the open (outer)
                            # edge toward the real threshold it borders,
                            # rather than a flat opaque block -- reads as
                            # "this category has no far boundary" instead
                            # of asserting a hard edge that isn't real
                            fade_steps = 8
                            neutral_color = "#EDE6D6"
                            slice_width = (x1 - x0) / fade_steps
                            for s in range(fade_steps):
                                sx0 = x0 + s * slice_width
                                sx1 = sx0 + slice_width
                                frac = (
                                    (s + 1) / fade_steps
                                    if k == 0
                                    else (fade_steps - s) / fade_steps
                                )
                                slice_alpha = strip_alpha * frac
                                if orientation == "vertical":
                                    ax_items.add_patch(
                                        Rectangle(
                                            (sx0, row - bar_half),
                                            sx1 - sx0,
                                            2 * bar_half,
                                            facecolor=neutral_color,
                                            alpha=slice_alpha,
                                            edgecolor="none",
                                        )
                                    )
                                else:
                                    ax_items.add_patch(
                                        Rectangle(
                                            (row - bar_half, sx0),
                                            2 * bar_half,
                                            sx1 - sx0,
                                            facecolor=neutral_color,
                                            alpha=slice_alpha,
                                            edgecolor="none",
                                        )
                                    )
                            # one clean outline over the whole segment,
                            # since the faded slices have none of their own
                            if orientation == "vertical":
                                ax_items.add_patch(
                                    Rectangle(
                                        (x0, row - bar_half),
                                        x1 - x0,
                                        2 * bar_half,
                                        facecolor="none",
                                        edgecolor="black",
                                        linewidth=0.5,
                                        alpha=strip_alpha,
                                    )
                                )
                            else:
                                ax_items.add_patch(
                                    Rectangle(
                                        (row - bar_half, x0),
                                        2 * bar_half,
                                        x1 - x0,
                                        facecolor="none",
                                        edgecolor="black",
                                        linewidth=0.5,
                                        alpha=strip_alpha,
                                    )
                                )
                        else:
                            color = category_color(k)
                            if orientation == "vertical":
                                ax_items.add_patch(
                                    Rectangle(
                                        (x0, row - bar_half),
                                        x1 - x0,
                                        2 * bar_half,
                                        facecolor=color,
                                        alpha=strip_alpha,
                                        edgecolor="black",
                                        linewidth=0.5,
                                    )
                                )
                            else:
                                ax_items.add_patch(
                                    Rectangle(
                                        (row - bar_half, x0),
                                        2 * bar_half,
                                        x1 - x0,
                                        facecolor=color,
                                        alpha=strip_alpha,
                                        edgecolor="black",
                                        linewidth=0.5,
                                    )
                                )

                        label_color = "dimgrey" if is_extreme else "black"
                        seg_center = (x0 + x1) / 2

                        if x1 - x0 > min_label_width:
                            if orientation == "vertical":
                                ax_items.text(
                                    seg_center,
                                    row,
                                    str(k),
                                    ha="center",
                                    va="center",
                                    fontsize=labelsize,
                                    fontweight="bold",
                                    color=label_color,
                                )
                            else:
                                # unrotated (unlike the item name labels,
                                # which run along the column) -- these sit
                                # inside a narrow column but read better
                                # upright than sideways. center_baseline
                                # (rather than center) reads as properly
                                # centred for digit-only text, which has no
                                # descenders for "center" to allow for.
                                ax_items.text(
                                    row,
                                    seg_center,
                                    str(k),
                                    ha="center",
                                    va="center_baseline",
                                    fontsize=labelsize,
                                    fontweight="bold",
                                    color=label_color,
                                )
                        else:
                            # too narrow for the number to sit inside its
                            # own segment -- draw it just outside instead,
                            # with a short leader connecting it back, using
                            # the inter-row gap so it doesn't intrude on
                            # the neighbouring row
                            # starts a third of the way into the bar
                            # itself (rather than right at its edge) so
                            # the leader visibly originates from the
                            # segment it's labelling
                            leader_len = bar_half * 0.35
                            leader_start = row + bar_half / 3
                            leader_end = row + bar_half + leader_len
                            if orientation == "vertical":
                                ax_items.plot(
                                    [seg_center, seg_center],
                                    [leader_start, leader_end],
                                    color=label_color,
                                    linewidth=0.6,
                                )
                                ax_items.text(
                                    seg_center,
                                    leader_end,
                                    str(k),
                                    ha="center",
                                    va="top",
                                    fontsize=labelsize * 0.75,
                                    fontweight="bold",
                                    color=label_color,
                                )
                            else:
                                ax_items.plot(
                                    [leader_start, leader_end],
                                    [seg_center, seg_center],
                                    color=label_color,
                                    linewidth=0.6,
                                )
                                ax_items.text(
                                    leader_end,
                                    seg_center,
                                    str(k),
                                    ha="left",
                                    va="center_baseline",
                                    fontsize=labelsize * 0.75,
                                    fontweight="bold",
                                    color=label_color,
                                )

                # one label per packed row, positioned like a tick label
                # just outside the axes, rather than inline after each
                # item's own bar (which put labels at wildly different
                # positions depending on where that item's own range fell)
                for label_row, label_text in row_labels.items():
                    row_disordered = any(
                        n in disordered_items for n in row_names[label_row]
                    )
                    label_kwargs = (
                        {"color": "firebrick", "fontweight": "bold"}
                        if row_disordered
                        else {"color": item_line_color}
                    )
                    if orientation == "vertical":
                        ax_items.text(
                            -0.015,
                            label_row,
                            f"{label_text} ",
                            transform=ax_items.get_yaxis_transform(),
                            ha="right",
                            va="center",
                            fontsize=labelsize,
                            fontfamily=item_font,
                            clip_on=False,
                            **label_kwargs,
                        )
                    else:
                        ax_items.text(
                            label_row,
                            -0.015,
                            f"{label_text} ",
                            transform=ax_items.get_xaxis_transform(),
                            ha="right",
                            va="center",
                            fontsize=labelsize,
                            fontfamily=item_font,
                            rotation=90,
                            rotation_mode="anchor",
                            clip_on=False,
                            **label_kwargs,
                        )

                if orientation == "vertical":
                    ax_items.spines["top"].set_visible(False)
                    ax_items.set_yticks([])
                    # ax's own bottom edge is an interior boundary (touches
                    # item_dist or ax_items), not a useful place for
                    # Location labels -- show a duplicate scale on ax's
                    # outer (top) edge instead, so a tall figure has
                    # Location ticks to read at both ends
                    ax.tick_params(
                        axis="x", bottom=False, top=True, labelbottom=False, labeltop=True
                    )
                else:
                    ax_items.spines["left"].set_visible(False)
                    ax_items.set_xticks([])
                    # ax is the leftmost panel here, so its default (left)
                    # tick side is already the figure's outer edge -- no
                    # repositioning needed, just leave it showing

            elif item_labels:
                # row 0 sits nearest the boundary with the persons panel,
                # deeper rows move further away -- inverted ylim for
                # vertical (row 0 at the top, closest to the persons panel
                # above) achieves that directly. Asymmetric margins: a
                # little breathing room near the x-axis boundary
                # (item_row_margin), and just enough at the panel's outer
                # edge (0.1) to keep the last row off the border.
                boundary_margin = item_row_margin
                if orientation == "vertical":
                    ax_items.set_ylim(max_depth + 0.1, -boundary_margin)
                else:
                    ax_items.set_xlim(-boundary_margin, max_depth + 0.1)

                for b, entries in item_groups.items():
                    x = bin_centers[b]
                    # sort by actual location, not name -- a real map
                    # orders items by where they fall
                    for i, (loc, name) in enumerate(sorted(entries)):
                        # rows 0..mark_row_depth-1 are the marks' own
                        # reserved band (see item_mark_rows) -- every
                        # item starts stacking uniformly above it
                        level = i + mark_row_depth
                        if orientation == "vertical":
                            # va='top' (not 'center') anchors every
                            # item's own near edge at its row, matching
                            # ha='left' below for horizontal -- centring
                            # let a short name like "Item_2" sit with a
                            # different-looking start than a long one
                            # like "Item_19" even at the identical row,
                            # since centring keeps the middle fixed while
                            # the two ends move with the string's length
                            ax_items.text(
                                x,
                                level,
                                str(name),
                                color=item_line_color,
                                rotation=90,
                                ha="center",
                                va="top",
                                fontsize=item_label_size,
                                fontfamily=item_font,
                            )
                        else:
                            # ha='left' (not 'center') anchors every
                            # item's own near edge at its row, rather
                            # than its centre -- see the vertical branch
                            # above for why
                            ax_items.text(
                                level,
                                x,
                                str(name),
                                color=item_line_color,
                                ha="left",
                                va="center",
                                fontsize=item_label_size,
                                fontfamily=item_font,
                            )

                # the item panel is a label area, not a real data axis --
                # strip its own ticks (no meaningful scale to read off),
                # and keep three of its four spines. The panel sits flush
                # against the persons panel (hspace/wspace=0), so its
                # near-side spine (top for vertical, left for horizontal)
                # would stack directly on top of the persons panel's own
                # boundary spine and the explicit axhline/axvline -- three
                # overlapping lines reading as one heavy, slightly blurred
                # one. Dropping the item panel's near-side spine leaves a
                # single clean boundary while still closing the other
                # three sides.
                if orientation == "vertical":
                    ax_items.spines["top"].set_visible(False)
                    ax_items.set_yticks([])
                    # ax's own bottom edge is an interior boundary (touches
                    # item_dist or ax_items), not a useful place for
                    # Location labels -- show a duplicate scale on ax's
                    # outer (top) edge instead, so a tall figure has
                    # Location ticks to read at both ends
                    ax.tick_params(
                        axis="x", bottom=False, top=True, labelbottom=False, labeltop=True
                    )
                else:
                    ax_items.spines["left"].set_visible(False)
                    ax_items.set_xticks([])
                    # ax is the leftmost panel here, so its default (left)
                    # tick side is already the figure's outer edge -- no
                    # repositioning needed, just leave it showing

            if distribution_markers:

                # freeze ax's own count-axis limits before drawing
                # anything into it, pre-expanded to match the "nice"
                # round tick matplotlib's own locator will pick for this
                # range -- that expansion normally happens much later
                # (the set_xticks call below, which snaps the outer edge
                # out to the locator's outermost tick, e.g. -77.7 -> -80)
                # well after this block has already computed and drawn
                # the marks. Replicating it here first means _in_per_unit
                # sees the axis's true final scale instead of a
                # provisional one that's about to change size under it --
                # marks are also, like any other artist, still their own
                # small source of autoscale drift once drawn (clip_on=
                # False), which this same freeze heads off
                if orientation == "vertical":
                    y0, y1 = ax.get_ylim()
                    nice_ticks = ax.get_yticks()
                    if len(nice_ticks):
                        y1 = max(y1, max(nice_ticks))
                    ax.set_ylim(y0, y1)
                else:
                    x0, x1 = ax.get_xlim()
                    nice_ticks = ax.get_xticks()
                    if len(nice_ticks):
                        x0 = min(x0, min(nice_ticks))
                    ax.set_xlim(x0, x1)

                def _mark_len(target_ax):
                    if orientation == "vertical":
                        _, y1 = target_ax.get_ylim()
                        return 0.015 * y1
                    else:
                        x0, _ = target_ax.get_xlim()
                        return 0.015 * abs(x0)

                # same size as the axis tick labels
                marker_fontsize = labelsize

                def _in_per_unit(target_ax):
                    # exact data-unit -> inch conversion for this axis.
                    # Built from the axes' own pixel bbox (fixed by the
                    # gridspec layout, stable regardless of xlim/ylim)
                    # together with x0/y1 -- the *stable* side of this
                    # axis's own extent, the same side _mark_len already
                    # reads. The other side (the boundary shared with the
                    # item panel) still gets pinned to exactly 0 further
                    # below, after distribution_markers runs, to correct
                    # for autoscale drift the marks' own overhang causes
                    # -- reading it here, before that pin, would use a
                    # transform that's about to change and mismatch
                    # whatever the item side's tick actually renders at
                    bbox = target_ax.get_window_extent(
                        renderer=fig.canvas.get_renderer()
                    )
                    if orientation == "vertical":
                        _, y1 = target_ax.get_ylim()
                        return (bbox.height / fig.dpi) / y1
                    else:
                        x0, _ = target_ax.get_xlim()
                        return (bbox.width / fig.dpi) / abs(x0)

                def add_distribution_markers(values, sign, color, target_ax, mark_len):
                    mean = values.mean()
                    sd = values.std()
                    marks = [
                        ("μ", mean),
                        ("μ−σ", mean - sd),
                        ("μ+σ", mean + sd),
                        ("μ−2σ", mean - 2 * sd),
                        ("μ+2σ", mean + 2 * sd),
                    ]
                    # fixed physical gap between the tick's own end and
                    # its label, matching the item side's mark_gap_in
                    # instead of a fraction of mark_len -- mark_len itself
                    # already varies with the count axis's own range, so
                    # multiplying it wouldn't give a gap of consistent
                    # physical size
                    gap = mark_gap_in / _in_per_unit(target_ax)
                    text_offset = sign * (mark_len + gap)

                    for label, loc in marks:
                        if orientation == "vertical":
                            target_ax.plot(
                                [loc, loc],
                                [0, sign * mark_len],
                                color=color,
                                linewidth=1,
                                # matplotlib's default line cap
                                # ("projecting") extends a line by half
                                # its own width past each endpoint -- with
                                # this tick's zorder placing it above the
                                # boundary line, that projection is what
                                # was visibly poking through to the other
                                # side, not the tick's own coordinates
                                solid_capstyle="butt",
                                clip_on=False,
                                zorder=6,
                            )
                            target_ax.text(
                                loc,
                                text_offset,
                                label,
                                color=color,
                                ha="center",
                                va="bottom" if sign > 0 else "top",
                                fontsize=marker_fontsize,
                                clip_on=False,
                                zorder=6,
                            )
                        else:
                            target_ax.plot(
                                [0, sign * mark_len],
                                [loc, loc],
                                color=color,
                                linewidth=1,
                                solid_capstyle="butt",
                                clip_on=False,
                                zorder=6,
                            )
                            target_ax.text(
                                text_offset,
                                loc,
                                label,
                                color=color,
                                ha="left" if sign > 0 else "right",
                                va="center",
                                fontsize=marker_fontsize,
                                clip_on=False,
                                zorder=6,
                            )

                # match the item side's own fixed physical tick length
                # when there's an item panel to match -- _mark_len's
                # fraction-of-count-range formula varies with whatever
                # dataset is plotted, so left alone it drifts arbitrarily
                # far from the item side's fixed length instead of
                # tracking it
                if item_mark_rows:
                    # matched to ax_items' own *realized* inches-per-row
                    # -- not row_height_in, since ax_items' actual
                    # xlim/ylim span is max_depth + 0.1 + item_row_margin,
                    # not just max_depth, so row_height_in alone slightly
                    # overstates how many inches one row really occupies.
                    # Uses the panel's full span (not _in_per_unit's
                    # stable-side-only logic -- that assumes an axis
                    # anchored at 0, which ax_items isn't); ax_items' own
                    # xlim/ylim is already final at this point, so there's
                    # no pending-pin instability to work around here
                    items_bbox = ax_items.get_window_extent(
                        renderer=fig.canvas.get_renderer()
                    )
                    if orientation == "vertical":
                        y0i, y1i = ax_items.get_ylim()
                        items_in_per_row = (items_bbox.height / fig.dpi) / abs(
                            y1i - y0i
                        )
                    else:
                        x0i, x1i = ax_items.get_xlim()
                        items_in_per_row = (items_bbox.width / fig.dpi) / abs(
                            x1i - x0i
                        )
                    person_mark_len = (
                        mark_tick_rows * items_in_per_row
                    ) / _in_per_unit(ax)
                else:
                    person_mark_len = _mark_len(ax)
                if group_by is None:
                    add_distribution_markers(
                        persons, person_sign, marker_color, ax, person_mark_len
                    )
                else:
                    # each group gets its own mean/SD marks in its own
                    # colour -- a single overall mark wouldn't mean much
                    # once the distribution's been split
                    for label, sub, fill_color, line_color in person_groups:
                        add_distribution_markers(
                            sub, person_sign, line_color, ax, person_mark_len
                        )

                # with its own item_distribution panel, item marks belong
                # on that panel's own scale (still flipped, matching the
                # hist/kde bars there) rather than overhanging into the
                # persons panel's boundary
                if item_distribution:
                    add_distribution_markers(
                        items,
                        item_sign,
                        marker_color,
                        ax_item_dist,
                        _mark_len(ax_item_dist),
                    )
                elif item_labels and not item_strip:
                    # drawn directly into the item panel's own row grid,
                    # all at row 0 (see item_mark_rows) -- rather than
                    # reaching in from ax's own count-axis scale. No item
                    # can ever share row 0, by construction, and every
                    # mark's own tick+label sits at exactly the same
                    # depth as every other, rather than varying with
                    # whatever bin it happens to land in. The tick itself
                    # is flush with the boundary shared with the persons
                    # panel (tick_near sits exactly at -item_row_margin,
                    # the item panel's own edge), matching the persons
                    # side's tick, which is flush with that same boundary
                    # from its own side.
                    tick_near = -item_row_margin
                    tick_far = tick_near + mark_tick_rows
                    for mark_label, mark_loc in item_mark_rows.items():
                        if orientation == "vertical":
                            ax_items.plot(
                                [mark_loc, mark_loc],
                                [tick_near, tick_far],
                                color=marker_color,
                                linewidth=1,
                                solid_capstyle="butt",
                                zorder=6,
                            )
                            # va='top' (not 'center') anchors every
                            # label's own near edge at mark_text_row,
                            # matching ha='left' below for horizontal --
                            # centring would let a short label like "mu"
                            # sit with more of a gap after the tick than
                            # a long one like "mu+2sigma"
                            ax_items.text(
                                mark_loc,
                                mark_text_row,
                                mark_label,
                                color=marker_color,
                                rotation=90,
                                ha="center",
                                va="top",
                                fontsize=marker_fontsize,
                            )
                        else:
                            ax_items.plot(
                                [tick_near, tick_far],
                                [mark_loc, mark_loc],
                                color=marker_color,
                                linewidth=1,
                                solid_capstyle="butt",
                                zorder=6,
                            )
                            # ha='left' (not 'center') anchors every
                            # label's own near edge at mark_text_row,
                            # rather than its centre -- centring would
                            # let a short label like "mu" sit with more
                            # of a gap after the tick than a long one
                            # like "mu+2sigma", which is exactly the
                            # inconsistent-looking start this is fixing
                            ax_items.text(
                                mark_text_row,
                                mark_loc,
                                mark_label,
                                color=marker_color,
                                ha="left",
                                va="center",
                                fontsize=marker_fontsize,
                            )
                else:
                    add_distribution_markers(
                        items,
                        -person_sign if item_labels else item_sign,
                        marker_color,
                        ax,
                        person_mark_len,
                    )

            # ax's and ax_item_dist's own Count/Density limits are set
            # explicitly further below (from person_range/item_range),
            # which fully supersedes pinning just one side of each to 0
            # here.

            padding = (hi - lo) * 0.05 if pad else 0
            loc_axis = ax_items if (item_labels and orientation == "vertical") else ax
            loc_axis_h = (
                ax_items if (item_labels and orientation == "horizontal") else ax
            )

            if item_labels and orientation == "horizontal":
                # ax_items is the rightmost panel, but a y-axis's ticks
                # default to its own left side -- an interior boundary
                # here, not the figure's outer edge. Move them (and the
                # "Location" title) to the right so the primary scale
                # lands at the outer edge, mirroring how ax_items' bottom
                # edge is already the outer edge in vertical orientation.
                ax_items.yaxis.set_ticks_position("right")
                ax_items.yaxis.set_label_position("right")

            # evenly spaced at tick_interval across the plotted range --
            # matplotlib's own automatic tick choice, unioned with the
            # exact data min/max (the previous approach), could land the
            # min/max on an irregular in-between spacing rather than the
            # regular grid the rest of the ticks follow
            def _regular_loc_ticks(axis_lo, axis_hi):
                start = np.ceil(axis_lo / tick_interval) * tick_interval
                n = int(np.floor((axis_hi - start) / tick_interval + 1e-9)) + 1
                return [start + i * tick_interval for i in range(max(n, 0))]

            if orientation == "vertical":
                ax.set_xlim(lo - padding, hi + padding)
                loc_ticks = _regular_loc_ticks(lo - padding, hi + padding)
                loc_axis.set_xticks(loc_ticks)
                if item_distribution:
                    # keeps this panel's gridlines aligned with the other
                    # two, even though its own tick labels stay hidden
                    ax_item_dist.set_xticks(loc_ticks)
            else:
                ax.set_ylim(lo - padding, hi + padding)
                loc_ticks = _regular_loc_ticks(lo - padding, hi + padding)
                loc_axis_h.set_yticks(loc_ticks)
                if item_distribution:
                    ax_item_dist.set_yticks(loc_ticks)

            if item_labels:
                # ax's own boundary spine is redundant with the explicit
                # axhline/axvline drawn further below -- both nominally
                # sit at data 0, but as different Artist types (Spine vs
                # Line2D) they can each round to a very slightly
                # different sub-pixel position even at the identical
                # data coordinate, leaving a faint second line just next
                # to the axhline/axvline rather than exactly under it.
                # Hiding the spine outright, rather than trying to
                # position it to coincide, removes that mismatch instead
                # of chasing sub-pixel alignment. Persons is always
                # flipped in horizontal orientation now, so the boundary
                # is always ax's right spine, never left.
                if orientation == "vertical":
                    ax.spines["bottom"].set_visible(False)
                else:
                    ax.spines["right"].set_visible(False)

            is_vertical = orientation == "vertical"

            # each panel's own (lo, hi) limits, set directly from
            # person_range/item_range (an explicit *_lim override, or
            # each distribution's own natural peak, resolved earlier)
            # rather than left to independent autoscale -- this is also
            # exactly what panel sizing above already assumed, so the
            # rendered axis and the physical space allocated for it
            # always agree. When persons and items still mirror onto the
            # same ax (item_labels=False), each of its own two sides
            # gets its own real range rather than a shared one.
            if item_labels:
                ax_lo = -person_range if person_sign < 0 else 0
                ax_hi = person_range if person_sign > 0 else 0
            else:
                item_extent = item_range * mirror_item_scale
                ax_lo = -(person_range if person_sign < 0 else item_extent)
                ax_hi = person_range if person_sign > 0 else item_extent
            if is_vertical:
                ax.set_ylim(ax_lo, ax_hi)
            else:
                ax.set_xlim(ax_lo, ax_hi)

            if item_distribution:
                dist_lo = -item_range if item_sign < 0 else 0
                dist_hi = item_range if item_sign > 0 else 0
                if is_vertical:
                    ax_item_dist.set_ylim(dist_lo, dist_hi)
                else:
                    ax_item_dist.set_xlim(dist_lo, dist_hi)

            def _side_ticks(reach, step):
                # evenly spaced ticks from 0 to reach inclusive, using
                # exactly the step already resolved (together with
                # reach itself) to divide it evenly -- generated
                # explicitly rather than left to a locator, which has no
                # reason to land on a divisor of this specific ceiling
                if reach <= 0 or step <= 0:
                    return [0.0]
                n = int(round(reach / step))
                return [i * step for i in range(n + 1)]

            def _relabel(
                ax_obj,
                pos_reach,
                pos_step,
                neg_reach,
                neg_step,
                drop_zero=False,
                pos_scale=1.0,
                neg_scale=1.0,
            ):
                # (tick position, label value) pairs: a side whose count
                # axis was stretched by mirror_item_scale keeps its tick
                # *labels* at the true counts, only spaced further apart
                # (pos_scale / neg_scale = 1 everywhere else).
                pairs = {(t * pos_scale, t) for t in _side_ticks(pos_reach, pos_step)}
                pairs |= {(-t * neg_scale, -t) for t in _side_ticks(neg_reach, neg_step)}
                pairs = sorted(pairs)
                # item_dist's own zero always sits exactly at the
                # boundary shared with persons -- in horizontal
                # orientation that boundary is a narrow vertical seam
                # with both panels' own "0" label sitting right next to
                # it, close enough to visually collide. Dropping
                # item_dist's own redundant zero (ax's own stays) avoids
                # that; not an issue in vertical orientation, where the
                # seam is horizontal and the two labels don't compete
                # for the same space.
                if drop_zero and not is_vertical:
                    pairs = [pl for pl in pairs if pl[0] != 0]
                positions = [p for p, _ in pairs]
                labels = (
                    [f"{abs(v):.2f}" for _, v in pairs]
                    if prop
                    else [str(int(round(abs(v)))) for _, v in pairs]
                )
                if is_vertical:
                    ax_obj.set_yticks(positions)
                    ax_obj.set_yticklabels(labels, fontsize=labelsize)
                else:
                    ax_obj.set_xticks(positions)
                    ax_obj.set_xticklabels(labels, fontsize=labelsize)

            if item_labels:
                if person_sign > 0:
                    _relabel(ax, person_range, person_step, 0, 1)
                else:
                    _relabel(ax, 0, 1, person_range, person_step)
            elif person_sign > 0:
                # vertical: items are the negative side of the shared axis
                _relabel(
                    ax,
                    person_range,
                    person_step,
                    item_range,
                    item_step,
                    neg_scale=mirror_item_scale,
                )
            else:
                # horizontal: items are the positive side of the shared axis
                _relabel(
                    ax,
                    item_range,
                    item_step,
                    person_range,
                    person_step,
                    pos_scale=mirror_item_scale,
                )

            if item_distribution:
                if item_sign > 0:
                    _relabel(ax_item_dist, item_range, item_step, 0, 1, drop_zero=True)
                else:
                    _relabel(ax_item_dist, 0, 1, item_range, item_step, drop_zero=True)

            # explicit boundary line at 0, replacing (not just visually
            # stacked on top of) the regular gridline that would
            # otherwise also be drawn there -- relying on this line's own
            # width/zorder to fully cover that gridline was fragile: as
            # two separately-rendered Line2D objects, sub-pixel rounding
            # can put them at very slightly different pixel positions
            # even at the same data coordinate 0, leaving a faint grey
            # sliver just next to the black line rather than under it.
            # Both this and the "hide the gridline that would otherwise
            # sit right under it" cleanup need the *final* tick set from
            # _relabel above -- doing this earlier (against whatever
            # ticks autoscale had chosen before the real limits were
            # even set) hid whichever gridline happened to occupy that
            # position in the stale tick list, not necessarily the one
            # actually at 0.
            if orientation == "vertical":
                ax.axhline(0, color="black", linewidth=1.3, zorder=5)
                for tick, gridline in zip(ax.get_yticks(), ax.yaxis.get_gridlines()):
                    if tick == 0:
                        gridline.set_visible(False)
                # ax's own axhline is clipped to ax's own box, so its
                # rendered width only ever eats into the persons side --
                # the item panel's side of that same boundary is left
                # with no black pixels to reach into at all, which is
                # what actually made the two sides look asymmetric (one
                # side's tick visibly "covers" part of the boundary,
                # the other's just abuts a boundary that was never
                # there on its side to begin with), even though both
                # ticks are the same true length. Mirroring the line on
                # ax_items, clipped to *its* own box, gives the item
                # side an equal, equally-covered sliver instead
                if item_labels:
                    ax_items.axhline(
                        -item_row_margin, color="black", linewidth=1.3, zorder=5
                    )
            else:
                ax.axvline(0, color="black", linewidth=1.3, zorder=5)
                if item_labels:
                    ax_items.axvline(
                        -item_row_margin, color="black", linewidth=1.3, zorder=5
                    )
                for tick, gridline in zip(ax.get_xticks(), ax.xaxis.get_gridlines()):
                    if tick == 0:
                        gridline.set_visible(False)

            if orientation == "vertical":
                loc_axis.set_xlabel(
                    "Location", fontsize=axis_font_size, fontweight="bold"
                )
                ax.set_ylabel(
                    "Density" if prop else "Count",
                    fontsize=axis_font_size,
                    fontweight="bold",
                )
                if item_distribution:
                    # one shared "Count"/"Density" label instead of one per
                    # panel -- re-centre ax's own label (rather than also
                    # labelling ax_item_dist) across both panels' combined
                    # span, computed in ax's own axes-fraction coordinates
                    # since item_dist sits directly below it. The x offset
                    # has to clear whichever panel's tick labels are wider.
                    # Measuring the tick labels' own rendered extent
                    # directly (rather than reading label.get_position()
                    # after a draw, which returns a display-space value
                    # from matplotlib's internal auto-layout transform,
                    # not a reusable axes-fraction one) and converting
                    # that through transAxes is what actually round-trips
                    # correctly into set_label_coords.
                    fig.canvas.draw()
                    renderer = fig.canvas.get_renderer()

                    def _leftmost_tick_x(target_ax):
                        return min(
                            t.get_window_extent(renderer=renderer).x0
                            for t in target_ax.yaxis.get_ticklabels()
                            if t.get_text()
                        )

                    left_px = min(_leftmost_tick_x(ax), _leftmost_tick_x(ax_item_dist))
                    label_x = ax.transAxes.inverted().transform((left_px, 0))[0]
                    # a little further out again, in axes-fraction, so the
                    # label doesn't sit flush against the tick numbers
                    label_x -= 0.03 * 0.6
                    mid_y = 0.5 * (1 - item_dist_panel_h / person_panel_h)
                    ax.yaxis.set_label_coords(label_x, mid_y)
            else:
                loc_axis_h.set_ylabel(
                    "Location", fontsize=axis_font_size, fontweight="bold"
                )
                ax.set_xlabel(
                    "Density" if prop else "Count",
                    fontsize=axis_font_size,
                    fontweight="bold",
                )
                if item_distribution:
                    # one shared "Count"/"Density" label instead of one per
                    # panel -- re-centre ax's own label across both panels'
                    # combined span (in ax's own axes-fraction coordinates,
                    # since item_dist sits directly to its right) instead
                    # of also labelling ax_item_dist. Measured the same way
                    # as the vertical case above, for the same reason:
                    # label.get_position() after a draw returns a display-
                    # space value from matplotlib's internal auto-layout
                    # transform, not one set_label_coords can reuse.
                    fig.canvas.draw()
                    renderer = fig.canvas.get_renderer()

                    def _bottommost_tick_y(target_ax):
                        return min(
                            t.get_window_extent(renderer=renderer).y0
                            for t in target_ax.xaxis.get_ticklabels()
                            if t.get_text()
                        )

                    bottom_px = min(_bottommost_tick_y(ax), _bottommost_tick_y(ax_item_dist))
                    label_y = ax.transAxes.inverted().transform((0, bottom_px))[1]
                    label_y -= 0.05 * 0.6
                    mid_x = 0.5 * (1 + item_dist_panel_w / person_panel_w)
                    ax.xaxis.set_label_coords(mid_x, label_y)

            ax.tick_params(axis="x", labelsize=labelsize)
            ax.tick_params(axis="y", labelsize=labelsize)
            if item_labels:
                loc_axis.tick_params(axis="x", labelsize=labelsize)
                loc_axis_h.tick_params(axis="y", labelsize=labelsize)
            if item_distribution:
                ax_item_dist.tick_params(axis="x", labelsize=labelsize)
                ax_item_dist.tick_params(axis="y", labelsize=labelsize)

            if title is not None:
                if item_labels:
                    # ax.set_title() centres over just the persons panel,
                    # which is only part of the figure width in horizontal
                    # orientation (reading as left-aligned) -- a figure-
                    # level suptitle centres over the whole figure instead,
                    # giving consistent placement in both orientations.
                    # Pulling y down from its 0.4-of-the-margin default
                    # leaves clearance below the title before the axes.
                    title_y = 1 - (margin_top_in * 0.4) / fig_h
                    fig.suptitle(
                        title, fontsize=title_font_size, fontweight="bold", y=title_y
                    )
                else:
                    ax.set_title(title, fontsize=title_font_size, fontweight="bold")

            # a legend can show handles from any axes, not just its own --
            # pooling persons' and item_dist's handles onto one legend
            # drawn on ax keeps a single legend box in the persons panel
            # instead of a second one competing for space (and fighting
            # "best" placement) inside item_dist
            legend_handles, legend_labels = ax.get_legend_handles_labels()
            if item_distribution:
                dist_handles, dist_labels = ax_item_dist.get_legend_handles_labels()
                legend_handles += dist_handles
                legend_labels += dist_labels
            # low-count locations sit near whichever side person_sign's
            # baseline is on -- normally that's the left (unflipped), so
            # "upper right" is clear. Flipped (horizontal + item_distribution)
            # puts the baseline on the right instead, so the empty corner
            # swaps to upper left.
            legend_loc = "upper left" if person_sign < 0 else "upper right"
            ax.legend(legend_handles, legend_labels, loc=legend_loc)

            if filename is not None:
                fig.savefig(filename + f".{file_format}", dpi=dpi)

            plt.show(block=False)
            plt.pause(0.001)
            plt.close(fig)
