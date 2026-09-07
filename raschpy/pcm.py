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


class PCM(Rasch):

    def __init__(
        self,
        responses,
        max_score_vector=None,
        extreme_persons=True,
        no_of_classes=5,
        validate=True,
        exogenous=None,
    ):
        """
        Initialise a Partial Credit Model object.

        Parameters
        ----------
        responses : pandas.DataFrame or PCM_Sim
            Response data with persons as rows and items as columns.
            Cell values should be integers in [0, max_score_vector[item]]
            or NaN for missing. Alternatively, pass a PCM_Sim object to
            instantiate directly from a simulation; generating parameters
            are stored in self.generating.
        max_score_vector : list of int or None, default None
            Maximum possible score for each item, in column order. If None,
            inferred from the observed data maximum per item. Must not be
            less than the observed maximum for any item.
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
        max_score_vector : pandas.Series
            Maximum possible score per item, indexed by item name.
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
        from raschpy.simulation.pcm_sim import PCM_Sim
        from raschpy.base import _SimParams

        if isinstance(responses, PCM_Sim):
            sim = responses
            self.generating = _SimParams()
            for attr, value in vars(sim).items():
                setattr(self.generating, attr, value)
            # max_score_vector is a raw list in the sim — build as Series
            self.max_score_vector = pd.Series(
                {
                    item: int(ms)
                    for item, ms in zip(sim.item_names, sim.max_score_vector)
                }
            )
            if max_score_vector is not None:
                passed = pd.Series(
                    {
                        item: int(ms)
                        for item, ms in zip(sim.item_names, max_score_vector)
                    }
                )
                mismatches = passed[passed != self.max_score_vector]
                if not mismatches.empty:
                    warnings.warn(
                        f"max_score_vector does not match sim.max_score_vector for items: "
                        f"{mismatches.index.tolist()}. Using passed max_score_vector."
                    )
                self.max_score_vector = passed
            responses = sim.responses
        else:
            if max_score_vector is None:
                self.max_score_vector = pd.Series(
                    {
                        item: int(max_score)
                        for item, max_score in zip(
                            responses.columns, responses.max().to_numpy()
                        )
                    }
                )
            else:
                self.max_score_vector = pd.Series(
                    {
                        item: int(max_score)
                        for item, max_score in zip(responses.columns, max_score_vector)
                    }
                )

        # Validate max_score_vector against observed data
        observed_max = responses.max()
        for item in self.max_score_vector.index:
            item_max = self.max_score_vector[item]
            item_obs = int(observed_max[item]) if not pd.isna(observed_max[item]) else 0
            if item_max < item_obs:
                raise ValueError(
                    f"max_score_vector[{item!r}]={item_max} is less than the maximum "
                    f"observed score ({item_obs}) for that item."
                )
            if item_max > item_obs:
                warnings.warn(
                    f"max_score_vector[{item!r}]={item_max} exceeds the maximum observed "
                    f"score ({item_obs}) for that item. Some score categories may be unobserved."
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
            max_scores = valid.notna().mul(self.max_score_vector, axis=1).sum(axis=1)
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

    """
    Partial Credit Model (Masters 1982) formulation of the polytomous Rasch model,
    with associated methods.
    """

    # ------------------------------------------------------------------
    # Core probability / expected-score functions (scalar, used in plots)
    # ------------------------------------------------------------------

    def cat_prob_centred(self, person_location, item_location, category, thresholds):
        """
        Compute the probability of a response category using centred parameterisation.

        Uses the PCM formulation with a central item location and
        Rasch-Andrich threshold offsets. Vectorised using numpy cumsum for
        performance. P(X=k) = exp(k*(b-d) - cumsum(tau)_k) / sum over all categories,
        where b is person location, d is item location, tau are thresholds (tau[0]=0 by convention).

        Parameters
        ----------
        person_location : float
            Person location estimate on the logit scale.
        item_location : float
            Central item location on the logit scale.
        category : int
            Response category (0 to max_score).
        thresholds : array-like
            Rasch-Andrich threshold offsets, length max_score + 1,
            centred at 0 per item.

        Returns
        -------
        float
            Probability of the specified category, in [0, 1].
        """
        max_score = len(thresholds)
        cats = np.arange(max_score + 1)
        cumsum = np.concatenate(([0.0], np.cumsum(thresholds)))
        log_nums = cats * (person_location - item_location) - cumsum
        log_nums -= log_nums.max()  # numerical stability
        nums = np.exp(log_nums)
        return nums[category] / nums.sum()

    def cat_prob_uncentred(self, person_location, category, thresholds):
        """
        Compute the probability of a response category using uncentred parameterisation.

        Uses the PCM formulation with uncentred (absolute) item-category thresholds.
        Numerically stabilised via log-sum-exp. P(X=k) = exp(k*b - cumsum(tau)_k) /
        sum over all categories, where b is person location and tau are uncentred thresholds.

        Parameters
        ----------
        person_location : float
            Person location estimate on the logit scale.
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
        log_nums = cats * person_location - cumsum
        log_nums -= log_nums.max()
        nums = np.exp(log_nums)
        return nums[category] / nums.sum()

    def exp_score_uncentred(self, person_location, thresholds):
        """
        Compute the expected score using uncentred threshold parameterisation.

        Calculates E[X | person location, thresholds] = sum(k * P(X=k)) over all
        categories, using uncentred threshold parameters.

        Parameters
        ----------
        person_location : float
            Person location estimate on the logit scale.
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
        log_nums = cats * person_location - cumsum
        log_nums -= log_nums.max()
        nums = np.exp(log_nums)
        return (cats * nums).sum() / nums.sum()

    def exp_score_centred(self, person_location, item_location, thresholds):
        """
        Compute the expected score using centred parameterisation.

        Calculates E[X | person location, item location, thresholds] using the centred
        PCM formulation with a central item location and Rasch-Andrich offsets.
        Numerically stabilised via log-sum-exp.

        Parameters
        ----------
        person_location : float
            Person location estimate on the logit scale.
        item_location : float
            Central item location on the logit scale.
        thresholds : array-like
            Rasch-Andrich threshold offsets, length max_score + 1.

        Returns
        -------
        float
            Expected score in [0, max_score].
        """
        thresh = np.asarray(thresholds)
        max_score = len(thresh)
        cats = np.arange(max_score + 1, dtype=float)
        cumsum = np.concatenate(([0.0], np.cumsum(thresh)))
        log_nums = cats * (person_location - item_location) - cumsum
        log_nums -= log_nums.max()
        nums = np.exp(log_nums)
        return (cats * nums).sum() / nums.sum()

    def variance_uncentred(self, person_location, thresholds):
        """
        Compute item variance (Fisher information) using uncentred parameterisation.

        Calculates Var[X | person location, thresholds] = sum((k - E[X])^2 * P(X=k)),
        equal to the Fisher information for the item at the given person location.
        Numerically stabilised via log-sum-exp.

        Parameters
        ----------
        person_location : float
            Person location estimate on the logit scale.
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
        log_nums = cats * person_location - cumsum
        log_nums -= log_nums.max()
        nums = np.exp(log_nums)
        probs = nums / nums.sum()
        expected = (cats * probs).sum()
        return ((cats - expected) ** 2 * probs).sum()

    def variance_centred(self, person_location, item_location, thresholds):
        """
        Compute item variance (Fisher information) using centred parameterisation.

        Calculates Var[X | person location, item location, thresholds] = sum((k - E[X])^2 * P(X=k)).
        Numerically stabilised via log-sum-exp.

        Parameters
        ----------
        person_location : float
            Person location estimate on the logit scale.
        item_location : float
            Central item location on the logit scale.
        thresholds : array-like
            Rasch-Andrich threshold offsets, length max_score + 1.

        Returns
        -------
        float
            Item variance / Fisher information.
        """
        thresh = np.asarray(thresholds)
        max_score = len(thresh)
        cats = np.arange(max_score + 1, dtype=float)
        cumsum = np.concatenate(([0.0], np.cumsum(thresh)))
        log_nums = cats * (person_location - item_location) - cumsum
        log_nums -= log_nums.max()
        nums = np.exp(log_nums)
        probs = nums / nums.sum()
        expected = (cats * probs).sum()
        return ((cats - expected) ** 2 * probs).sum()

    def kurtosis_uncentred(self, person_location, thresholds):
        """
        Compute the fourth central moment of the response distribution (uncentred).

        Calculates sum((k - E[X])^4 * P(X=k)) using uncentred threshold
        parameterisation. Used in the Wilson-Hilferty approximation for
        standardised fit statistics (Infit Z, Outfit Z).

        Parameters
        ----------
        person_location : float
            Person location estimate on the logit scale.
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
        log_nums = cats * person_location - cumsum
        log_nums -= log_nums.max()
        nums = np.exp(log_nums)
        probs = nums / nums.sum()
        expected = (cats * probs).sum()
        return ((cats - expected) ** 4 * probs).sum()

    def kurtosis_centred(self, person_location, item_location, thresholds):
        """
        Compute the fourth central moment of the response distribution (centred).

        Calculates sum((k - E[X])^4 * P(X=k)) using centred PCM parameterisation.

        Parameters
        ----------
        person_location : float
            Person location estimate on the logit scale.
        item_location : float
            Central item location on the logit scale.
        thresholds : array-like
            Rasch-Andrich threshold offsets, length max_score + 1.

        Returns
        -------
        float
            Fourth central moment of the response distribution.
        """
        thresh = np.asarray(thresholds)
        max_score = len(thresh)
        cats = np.arange(max_score + 1, dtype=float)
        cumsum = np.concatenate(([0.0], np.cumsum(thresh)))
        log_nums = cats * (person_location - item_location) - cumsum
        log_nums -= log_nums.max()
        nums = np.exp(log_nums)
        probs = nums / nums.sum()
        expected = (cats * probs).sum()
        return ((cats - expected) ** 4 * probs).sum()

    # ------------------------------------------------------------------
    # Vectorised cat-probability engine (core internal workhorse)
    # ------------------------------------------------------------------

    def _cat_probs_matrix(self, person_locations, items, thresholds=None):
        """
        Vectorised category probability computation used by person(), warm(),
        fit_statistics(), csem(), and person_lookup_table().

        Replaces five copies of the nested  "for item / for cat / sum()"  loop.

        Returns
        -------
        probs : ndarray, shape (max_max_score+1, N, n_items)
            probs[cat, person_idx, item_idx] = P(X=cat | person location, item)
            Categories beyond an item's max_score are set to 0.
        cats_arr : ndarray, shape (max_max_score+1,)
        """
        if thresholds is None:
            thresholds = self.thresholds_uncentred

        ab = np.asarray(person_locations, dtype=float)  # (N,)
        n = len(ab)
        n_items = len(items)
        max_max_score = int(max(len(thresholds.loc[it].dropna()) for it in items))
        cats_arr = np.arange(max_max_score + 1, dtype=float)  # (C,)

        log_probs = np.full((max_max_score + 1, n, n_items), -np.inf)

        for j, item in enumerate(items):
            thresh = thresholds.loc[item].dropna().values.astype(float)  # (m,)
            m = len(thresh)
            # prefix sums: cumsum[k] = sum(thresh[:k])
            cumsum = np.concatenate(([0.0], np.cumsum(thresh)))  # (m+1,)
            # log numerator for category k: k*person location - cumsum[k]
            # shape: (m+1, N)
            log_num = cats_arr[: m + 1, None] * ab[None, :] - cumsum[:, None]
            log_probs[: m + 1, :, j] = log_num

        # Numerically stable softmax along category axis
        log_max = np.max(log_probs, axis=0, keepdims=True)  # (1, N, n_items)
        with np.errstate(invalid="ignore"):
            probs = np.exp(log_probs - log_max)
        probs[~np.isfinite(log_probs)] = 0.0
        probs /= probs.sum(axis=0, keepdims=True)

        return probs, cats_arr

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def _build_pairwise_matrix(self):
        """
        Raw (unsmoothed) directed pairwise comparison matrix used by
        calibrate() and check_data_connectivity(). One row/column per
        item-category (not per item): entry (item_1's category i+1,
        item_2's category j) counts persons who scored category i+1 on
        item_1 and category j on item_2.

        Same-item (i == j) cells are forced to zero. PAIR compares pairs of
        distinct items; a same-item cell can never be a real comparison (a
        person has exactly one score per item, so two different categories
        of the same item can't both be true for them). The adjacent-category
        cells (threshold_k vs threshold_k+1 of the same item) still end up
        numerically populated by the row/col one-hot construction below, but
        that's the marginal frequency of category k relabelled as a
        pairwise cell, not a genuine comparison -- including it measurably
        biases both item and threshold recovery upward (beta > 1 in
        simulation) and masks real signal from mixed max_score_vector
        designs, where it also structurally leaves the non-adjacent
        same-item cells at zero regardless of sample size (see
        raschpy_pcm_within_item_bias memory note).

        Returns
        -------
        matrix : numpy.ndarray, shape (D, D), D = sum(max_score_vector)
        row_items : numpy.ndarray, length D
            Item name owning each row/column (repeated per category).
        """
        df_array = self.responses.to_numpy()
        row_blocks = []
        col_blocks = []
        row_items = []

        for item in range(self.no_of_items):
            max_k = self.max_score_vector.iloc[item]
            row_items.extend([self.item_names[item]] * max_k)

            scores = df_array[:, item]
            onehot = np.zeros((df_array.shape[0], max_k + 1), dtype=np.float64)
            valid = ~np.isnan(scores)
            onehot[valid, scores[valid].astype(int)] = 1.0
            row_blocks.append(onehot[:, 1:])  # categories 1..max_k
            col_blocks.append(onehot[:, :max_k])  # categories 0..max_k-1

        row_matrix = np.concatenate(row_blocks, axis=1)
        col_matrix = np.concatenate(col_blocks, axis=1)
        matrix = row_matrix.T @ col_matrix

        row_items = np.array(row_items)
        same_item = row_items[:, None] == row_items[None, :]
        matrix[same_item] = 0.0

        return matrix, row_items

    def _pcm_two_pass_solve(self, matrix, constant, matrix_power):
        """Shared densify-and-solve step for one pass of the two-pass rule.

        `constant` may be a scalar (pass 1) or a per-pair matrix (pass 2).
        Always densifies via matrix powering (matrix_power>=1) rather than
        'log-lik'-style direct zero-fill -- the two-pass taper is only
        validated, and only wins, against a densified matrix (see
        _pcm_two_pass_matrix's docstring).
        """
        constant_matrix = ((matrix + matrix.T) > 0).astype(np.float64) * constant
        mat_c = matrix + constant_matrix
        diag_add = np.diagonal(constant) if np.ndim(constant) else constant
        np.fill_diagonal(mat_c, mat_c.diagonal() + diag_add)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            mat = np.linalg.matrix_power(mat_c, matrix_power)
            mat_pow = matrix_power
            fallback = constant if not np.ndim(constant) else np.min(constant)
            while 0 in mat:
                mat = mat @ mat_c
                mat_pow += 1
                if mat_pow == matrix_power + 5:
                    mat = mat + fallback
                    break
        return mat

    def _pcm_two_pass_matrix(self, matrix, matrix_power=None):
        """
        Two-pass smoothing for calibrate(robust=True).

        Pass 1 solves at a scalar constant derived from the mean
        co-observation count, max_score, and missing fraction. Pass 2
        re-solves with a per-pair constant tapered down for item-threshold
        pairs far apart in the pass-1 estimate -- the near-Guttman cross-item
        pairs (low threshold of an easy item vs. high threshold of a hard
        one) individually carry almost no information but collectively
        anchor the scale; a flat large constant on them manufactures
        spurious ties and compresses the estimated spread, so the taper lets
        a big base constant clean up the informative near/within-item pairs
        without that compression.

        A flat constant=0.1 matches or beats this two-pass construction
        almost everywhere once _build_pairwise_matrix() excludes same-item
        cells (see that method's docstring) -- this is reserved for the one
        corner it still protects: missing >= ~0.6 combined with max_score >=
        ~5, where per-pair counts get too thin for a flat constant to serve
        well. Both passes solve via 'cos' on a densified (matrix_power>=1)
        matrix, not 'log-lik' -- checked directly against 'log-lik' in that
        target corner, 'cos' wins in every condition tested, sometimes by
        30%+ (a densified matrix apparently matters more than which
        priority-vector method reads it, once the comparison graph is this
        thin). calibrate()'s own `method` argument is ignored for this path.

        Returns
        -------
        mat : numpy.ndarray
            Pass-2 smoothed matrix, ready for priority_vector().
        c1 : float
            The pass-1 scalar constant (recorded as self.constant).
        """
        off = ~np.eye(matrix.shape[0], dtype=bool)
        meancount = float((matrix + matrix.T)[off].mean()) if off.any() else 0.0
        missing = float(self.responses.isna().mean().mean())
        K = int(self.max_score_vector.max())
        blend = float(np.clip(0.0865 * K**-1.72 * meancount * np.exp(1.27 * missing), 0.02, 60.0))
        c1 = 2.5 * blend
        # Densification plateaus at power 5 -- 3 captures most of the gain, 5
        # captures the rest, 6+ is bit-identical to 5 (see the power sweep in
        # raschpy_pcm_within_item_bias). Start at 5 rather than the usual 3.
        mp = matrix_power if matrix_power else 5

        mat1 = self._pcm_two_pass_solve(matrix, c1, mp)
        dhat = self.priority_vector(mat1, method="cos", pcm=True)
        dhat = dhat.to_numpy() if hasattr(dhat, "to_numpy") else np.array(dhat)

        gap = np.abs(dhat[:, None] - dhat[None, :])
        h = -max(0.6 / K, 0.1)
        c_ab = np.maximum(c1 * (1 + h * gap), 0.02)

        mat2 = self._pcm_two_pass_solve(matrix, c_ab, mp)

        return mat2, c1

    def calibrate(
        self, constant=None, method="log-lik", matrix_power=None, log_lik_tol=0.000001,
        robust=False,
    ):
        """
        Estimate item thresholds using the PAIR (Pairwise) algorithm.

        Constructs a joint score-category frequency matrix across all item
        pairs and threshold combinations using vectorised operations, then
        raises it to successive powers to resolve structural zeroes (Choppin's
        matrix power property). A priority vector is extracted from the resolved
        matrix to obtain uncentred threshold estimates. Central item locations
        are derived as the mean of each item's uncentred thresholds, and centred
        thresholds are computed as deviations from this mean.

        Issues a UserWarning if only one item is present, or if constant=0
        and any item has all-maximum scores.

        Parameters
        ----------
        constant : float or None, default None
            Additive smoothing constant applied to the frequency matrix.
            ``None`` resolves to 0.1, unless ``robust=True`` (see below). An
            explicit value always takes precedence over ``robust``. Use 0 to
            disable smoothing; estimation may fail if any item has
            all-maximum or all-minimum scores.
        method : str, default 'log-lik'
            Priority vector extraction method. See base.priority_vector().
        matrix_power : int or None, default None
            Initial matrix power before checking for structural zeroes.
            None resolves to 0 for method='log-lik' (no powering), else 5.
        log_lik_tol : float, default 0.000001
            Log-likelihood convergence tolerance for priority vector extraction.
        robust : bool, default False
            When ``True`` and ``constant`` is ``None``, use a two-pass
            smoothing construction instead of the flat 0.1 default: pass 1
            solves at a scalar constant derived from mean co-observation
            count, max_score, and missing fraction; pass 2 re-solves with a
            per-pair constant tapered down for item-threshold pairs far
            apart in the pass-1 estimate (see _pcm_two_pass_matrix). A flat
            constant=0.1 matches or beats this everywhere in simulation
            except one corner -- missing >= ~0.6 combined with max_score >=
            ~5 -- where per-pair counts get too thin for a flat constant;
            two-pass is reserved for protecting that corner, not a general
            improvement. Ignored, with a UserWarning, if ``constant`` is
            also given. Method is fixed to 'cos' internally for this path
            regardless of ``method`` -- checked directly against 'log-lik'
            in the target corner, 'cos' on a densified matrix wins in every
            condition tested, sometimes by 30%+.

        Attributes set
        --------------
        thresholds_uncentred : dict
            {item: numpy.ndarray} of uncentred threshold estimates per item.
        items : pandas.Series
            Central item location (mean of uncentred thresholds) per item.
        thresholds_centred : dict
            {item: numpy.ndarray} of centred threshold offsets per item.
        threshold_list : numpy.ndarray
            Flat array of all uncentred thresholds concatenated.
        constant : float
            The additive smoothing constant actually used, after resolving
            ``None`` / ``robust`` (the pass-1 scalar when ``robust=True``).
        null_persons : pandas.Index
            Persons dropped prior to calibration due to entirely missing data.
        """

        if len(self.responses.columns) == 1:
            warnings.warn(
                "Only one item detected. This model with a single item reduces to RSM "
                "with raters as items. Consider reconfiguring and using RSM instead.",
                UserWarning,
                stacklevel=2,
            )

        if constant is not None and robust:
            warnings.warn(
                f"Both `constant={constant}` and `robust=True` were passed. An "
                f"explicit `constant` takes precedence: `constant={constant}` "
                f"will be used and the two-pass robust rule will not be applied. "
                f"To use the rule, pass `robust=True` with `constant=None`.",
                UserWarning,
                stacklevel=2,
            )

        resolved_constant = 0.1 if constant is None else constant
        use_two_pass = constant is None and robust

        if resolved_constant == 0 and not use_two_pass:
            # Unsmoothed PAIR: re-run the structural connectivity check on the
            # data as it stands now (not just at __init__(validate=True)) --
            # constant=0 removes the only thing masking a directionally
            # isolated item or disconnected component, both of which
            # check_data_connectivity() already detects and warns about
            # (does not raise; 'log-lik' tolerates an incomplete comparison
            # graph and still returns an answer for the rest of the items).
            self.check_data_connectivity()

        all_null_mask = self.responses.isnull().all(axis=1)
        self.null_persons = self.responses.index[all_null_mask]
        if all_null_mask.any():
            self.responses = self.responses.loc[~all_null_mask]
        self.no_of_persons = self.responses.shape[0]

        matrix, _ = self._build_pairwise_matrix()
        cum_scores = np.concatenate(([0], np.cumsum(self.max_score_vector.to_numpy())))

        if use_two_pass:
            mat, resolved_constant = self._pcm_two_pass_matrix(matrix, matrix_power=matrix_power)
            method = "cos"
        else:
            constant = resolved_constant
            matrix_power = self._resolve_matrix_power(method, matrix_power)

            constant_matrix = ((matrix + matrix.T) > 0).astype(np.float64) * constant
            matrix = matrix + constant_matrix
            np.fill_diagonal(matrix, matrix.diagonal() + constant)

            # Sparse/disconnected resamples can blow this up to inf/nan before the zero-check
            # loop below terminates; not a real numerical error, so suppress the noise.
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                if matrix_power == 0:
                    # No powering: 'log-lik' (Bradley-Terry) consumes structural
                    # zeroes directly. Fill remaining off-diagonal zeroes
                    # (unobserved threshold pairs) with `constant`.
                    mat = np.array(matrix, dtype=np.float64)
                    if constant:
                        off_diagonal_mask = ~np.eye(matrix.shape[0], dtype=bool)
                        mat[off_diagonal_mask & (mat == 0)] = constant
                else:
                    mat = np.linalg.matrix_power(matrix, matrix_power)
                    mat_pow = matrix_power
                    while 0 in mat:
                        mat = mat @ matrix
                        mat_pow += 1
                        if mat_pow == matrix_power + 5:
                            mat += constant
                            break

        threshold_vector = self.priority_vector(
            mat, method=method, log_lik_tol=log_lik_tol, pcm=True
        )
        self.threshold_list = threshold_vector
        self.constant = resolved_constant

        split_indices = cum_scores[1:-1]
        threshold_vector_np = (
            threshold_vector.to_numpy()
            if hasattr(threshold_vector, "to_numpy")
            else np.array(threshold_vector)
        )
        item_threshold_arrays = np.split(threshold_vector_np, split_indices)

        items_dict = {}
        thresholds_uncentred_dict = {}
        thresholds_dict = {}

        for i, item in enumerate(self.responses.columns):
            uncentred = item_threshold_arrays[i]
            item_mean = np.mean(uncentred)
            thresholds_uncentred_dict[item] = pd.Series(uncentred)
            items_dict[item] = item_mean
            centered = uncentred - item_mean
            thresholds_dict[item] = centered

        self.items = pd.Series(items_dict)

        # Store as NaN-padded DataFrames (items × thresholds)
        self.thresholds_uncentred = pd.DataFrame(
            {
                item: pd.Series(thresholds_uncentred_dict[item])
                for item in self.responses.columns
            }
        ).T
        self.thresholds_uncentred.columns = range(1, self.thresholds_uncentred.shape[1] + 1)

        max_len = max(len(thresholds_dict[item]) for item in self.responses.columns)
        thr_rows = {}
        for item in self.responses.columns:
            arr = thresholds_dict[item]
            row = np.full(max_len, np.nan)
            row[: len(arr)] = arr
            thr_rows[item] = row
        self.thresholds = pd.DataFrame(thr_rows).T
        self.thresholds.columns = range(1, self.thresholds.shape[1] + 1)

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
        threshold_check="both",
        alpha=0.05,
        correction="bh",
        warm_corr=True,
        tolerance=0.00001,
        max_iters=100,
        ext_score_adjustment=0.5,
        constant=None,
        method="log-lik",
        robust=False,
        matrix_power=None,
        log_lik_tol=0.000001,
        seed=None,
    ):
        """
        Anchor item location estimates onto externally-supplied values.

        Supports item banking: calibrates this dataset's own item
        locations as usual, then shifts the item-location scale by a
        translation constant so that a subset of common ("anchor") items
        line up with externally-supplied reference locations (e.g. from
        a bank of previously-calibrated items). This is a translation only
        (PCM item discrimination is fixed at 1 across items), exactly as
        in SLM/RSM.

        Each item's own centred threshold offsets (self.thresholds) are
        left untouched by anchoring itself — unlike RSM, PCM has no single
        shared threshold vector to speak of in the first place: every item
        already has its own category structure. Translating item
        location doesn't need to touch it either way.

        Optionally (anchor_thresholds=, threshold_check='both' by default),
        if you also have externally-supplied reference threshold structures
        for some of the anchor items (e.g. the bank's own per-item category
        structure — not every bank publishes this, so anchor_thresholds
        need not cover every anchor item), this checks whether they're
        actually compatible with this dataset via likelihood-ratio tests:
        holding that item's location fixed at its anchored value, compare
        the model fit using this dataset's own freely-estimated threshold
        offsets for that item against using the given ones instead
        (person locations re-estimated over the whole test in each case). Because
        PCM thresholds are inherently per-item, this can be run per item
        (threshold_check='per_item', one independent test per supplied
        item, BH/Bonferroni-corrected for multiple comparisons), as a
        single combined test across every supplied item at once
        (threshold_check='aggregate'), both (default), or skipped entirely
        (threshold_check='none'). (As with RSM, AIC/BIC don't apply here —
        both candidates are point-values in the same fixed-length model, so
        there's no real parameter-count difference to penalise for; only
        LR's fixed-vs-free chi-squared framing is valid.)

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
            Externally-supplied reference locations, keyed/indexed by
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
            externally-supplied value from anchors, unchanged.
            'rejected': anchors kept by selection still keep their exact
            supplied value, but anchors rejected as outliers instead
            receive their own shifted, freshly-calibrated value. Requires
            selection_method in ('robust_z', 'wald') and no adj override;
            otherwise falls back to 'none' with a warning.
            'all': every anchor item receives its own shifted value like
            any other item, overwriting the supplied anchor value with
            what this dataset actually observed.
        plot : bool, default True
            If True and selection_method in ('robust_z', 'wald') (so a
            selection table exists), calls plot_anchor_selection()
            automatically at the end. Has no effect when
            selection_method='none' or adj is supplied directly.
        plot_kwargs : dict or None, default None
            Extra keyword arguments forwarded to plot_anchor_selection()
            when plot=True (e.g. filename, xmin/xmax, title).
        anchor_thresholds : dict or None, default None
            {item_name: array-like} of externally-supplied reference
            threshold offsets (same length/shape as self.thresholds.loc
            [item].dropna() — i.e. that item's own centred category-
            structure shape, not an absolute/uncentred value). Only items
            present in this dict are checked; need not cover every anchor
            item. If None, the threshold-structure check is skipped
            entirely regardless of threshold_check.
        threshold_check : {'per_item', 'aggregate', 'both', 'none'}, default 'both'
            'per_item': one independent LR test per item in anchor_thresholds
            (df = that item's max_score - 1 each), correction-adjusted.
            'aggregate': one combined LR test swapping every supplied item's
            thresholds in simultaneously (df = sum of each item's
            max_score - 1), no multiple-comparison correction needed since
            it's a single test.
            'both': runs both (default) — the per-item table for
            localisation, the aggregate test for an overall answer.
            'none': skips the threshold-structure check entirely, even if
            anchor_thresholds is supplied.
        alpha : float, default 0.05
            Significance level for all LR tests (compared against the
            corrected p-value for per-item tests, raw p for the aggregate
            test).
        correction : {'bh', 'bonferroni', None}, default 'bh'
            Multiple-comparison correction across the per-item tests only
            (mirrors dif_test's convention elsewhere in the package). None
            uses raw p-values, uncorrected. Ignored for the aggregate test.
        warm_corr, tolerance, max_iters, ext_score_adjustment :
            Person-estimation kwargs, used only by the threshold-structure
            comparison (threshold_check != 'none') (person locations must
            be re-estimated under each candidate threshold table before
            the log-likelihoods are comparable).
        constant, method, matrix_power, log_lik_tol : floats
            Calibration kwargs, used only if calibrate is triggered.

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
            correlation and SD ratio before/after trimming, and the
            translation constant applied.
        anchor_plot : matplotlib.figure.Figure or None
            The figure from the auto-triggered plot_anchor_selection()
            call. None if plot=False, selection_method='none', or adj was
            supplied directly.
        anchor_threshold_test : pandas.DataFrame or None
            Per-item LR test table (LL_estimated, LL_given, LR, df, p,
            p_corrected, Flagged), indexed by item. None if
            anchor_thresholds not supplied, threshold_check='none', or
            threshold_check='aggregate' only.
        anchor_threshold_test_aggregate : pandas.Series or None
            Combined LR test across every item in anchor_thresholds at
            once (LL_estimated, LL_given, LR, df, p, Flagged). None if
            anchor_thresholds not supplied, threshold_check='none', or
            threshold_check='per_item' only.
        calibrate_anchor_runs : dict
            Every call's results above, keyed by tuple(sorted(anchors.
            items())), so results from an earlier anchors call survive a
            later call with a different anchor set instead of being
            overwritten.
        """
        if overwrite_anchors not in ("none", "rejected", "all"):
            raise ValueError("overwrite_anchors must be 'none', 'rejected', or 'all'")
        if threshold_check not in ("per_item", "aggregate", "both", "none"):
            raise ValueError(
                "threshold_check must be 'per_item', 'aggregate', 'both', or 'none'"
            )
        if correction not in ("bh", "bonferroni", None):
            raise ValueError("correction must be 'bh', 'bonferroni', or None")
        if selection_method not in ("robust_z", "wald", "none"):
            raise ValueError("selection_method must be 'robust_z', 'wald', or 'none'")

        if not isinstance(anchors, pd.Series):
            anchors = pd.Series(anchors)

        if calibrate or not hasattr(self, "items"):
            self.calibrate(
                constant=constant,
                robust=robust,
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
                    robust=robust,
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

        if threshold_check != "none" and anchor_thresholds:
            for item, offsets in anchor_thresholds.items():
                expected_len = int(self.max_score_vector[item])
                if len(offsets) != expected_len:
                    raise ValueError(
                        f"anchor_thresholds[{item!r}] has length "
                        f"{len(offsets)}, expected {expected_len} "
                        f"(this item's max_score)."
                    )

            pe_kw = dict(
                warm_corr=warm_corr,
                tolerance=tolerance,
                max_iters=max_iters,
                ext_score_adjustment=ext_score_adjustment,
            )
            # Baseline (H1): every item at its own anchored location +
            # freely-estimated centred thresholds — identical across every
            # per-item test and the aggregate test, so compute it once.
            anchored_uncentred = self.thresholds.add(self.anchor_items, axis=0)
            ll_estimated = self._threshold_structure_ll(anchored_uncentred, **pe_kw)

            def _swapped(items_and_offsets):
                swapped = anchored_uncentred.copy()
                width = swapped.shape[1]
                for it, offs in items_and_offsets.items():
                    offs = np.asarray(offs, dtype=float)
                    row = np.full(width, np.nan)
                    row[: len(offs)] = self.anchor_items[it] + offs
                    swapped.loc[it] = row
                return swapped

            if threshold_check in ("per_item", "both"):
                rows = {}
                for item, offsets in anchor_thresholds.items():
                    df_i = len(offsets) - 1
                    ll_given_i = self._threshold_structure_ll(
                        _swapped({item: offsets}), **pe_kw
                    )
                    lr_i = max(0.0, 2 * (ll_estimated - ll_given_i))
                    p_i = float(chi2.sf(lr_i, df_i))
                    rows[item] = {
                        "LL_estimated": ll_estimated,
                        "LL_given": ll_given_i,
                        "LR": lr_i,
                        "df": df_i,
                        "p": p_i,
                    }
                table = pd.DataFrame(rows).T
                if correction == "bh":
                    table["p_corrected"] = self._bh_correction(table["p"])
                elif correction == "bonferroni":
                    table["p_corrected"] = (table["p"] * len(table)).clip(upper=1)
                p_col = "p_corrected" if correction else "p"
                table["Flagged"] = table[p_col] < alpha
                self.anchor_threshold_test = table

                if table["Flagged"].any():
                    warnings.warn(
                        "anchor_thresholds differs substantially from this "
                        "dataset's own estimated threshold structure for "
                        f"{list(table.index[table['Flagged']])} (per-item LR "
                        "test; see anchor_threshold_test). The rating-scale "
                        "category structure may not actually be shared with "
                        "the anchor source for these items — translation-"
                        "only equating via anchor_items assumes it is.",
                        UserWarning,
                        stacklevel=2,
                    )
            else:
                self.anchor_threshold_test = None

            if threshold_check in ("aggregate", "both"):
                df_agg = sum(len(offs) - 1 for offs in anchor_thresholds.values())
                ll_given_agg = self._threshold_structure_ll(
                    _swapped(anchor_thresholds), **pe_kw
                )
                lr_agg = max(0.0, 2 * (ll_estimated - ll_given_agg))
                p_agg = float(chi2.sf(lr_agg, df_agg))
                flagged_agg = p_agg < alpha
                self.anchor_threshold_test_aggregate = pd.Series(
                    {
                        "LL_estimated": ll_estimated,
                        "LL_given": ll_given_agg,
                        "LR": lr_agg,
                        "df": df_agg,
                        "p": p_agg,
                        "Flagged": flagged_agg,
                    },
                    name="Anchor threshold structure test (aggregate)",
                )
                if flagged_agg:
                    warnings.warn(
                        "anchor_thresholds differs substantially from this "
                        "dataset's own estimated threshold structure, "
                        "combined across all supplied items (aggregate LR "
                        "test; see anchor_threshold_test_aggregate). The "
                        "rating-scale category structure may not actually "
                        "be shared with the anchor source — translation-"
                        "only equating via anchor_items assumes it is.",
                        UserWarning,
                        stacklevel=2,
                    )
            else:
                self.anchor_threshold_test_aggregate = None
        else:
            self.anchor_threshold_test = None
            self.anchor_threshold_test_aggregate = None

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
            anchor_threshold_test_aggregate=self.anchor_threshold_test_aggregate,
        )

    def _threshold_structure_ll(
        self, thresholds_uncentred, warm_corr=True, tolerance=0.00001,
        max_iters=100, ext_score_adjustment=0.5,
    ):
        """
        Log-likelihood of this dataset's responses using a candidate
        uncentred (absolute) threshold table, with person locations re-estimated
        under that specific parameter set.

        Used by calibrate_anchor's threshold_check diagnostic to compare
        candidate threshold structures on a fair footing — person locations can't
        be reused across candidates since they depend on the threshold
        table too. Uses a scratch PCM instance (same convention as
        std_errors' bootstrap and andersen_lr_test's group refits) so
        self's own calibration state is never touched.
        """
        probe = PCM(self.responses, self.max_score_vector)
        probe.thresholds_uncentred = thresholds_uncentred
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
        constant=None,
        method="log-lik",
        robust=False,
        matrix_power=None,
        log_lik_tol=0.000001,
        no_of_samples=500,
        seed=None,
    ):
        """
        Estimate bootstrap standard errors for item threshold estimates.

        Draws no_of_samples bootstrap resamples of person-level response data,
        calibrates each resample, and computes the standard deviation of
        threshold and central item location estimates across samples. Also
        computes category width SEs (SE of spacing between adjacent thresholds).

        Parameters
        ----------
        interval : float or None, default None
            Confidence interval width (e.g. 0.95). If None, only SEs computed.
        constant : float, default 0.1
            Additive smoothing constant for bootstrap calibrations.
        method : str, default 'log-lik'
            Priority vector extraction method.
        matrix_power : int or None, default None
            Matrix power for bootstrap calibrations. None resolves to 0 for
            method='log-lik' (no powering), else 5.
        log_lik_tol : float, default 0.000001
            Convergence tolerance for bootstrap calibrations.
        no_of_samples : int, default 500
            Number of bootstrap resamples.
        seed : int or None, default None
            Seed for the bootstrap resampling RNG. Pass an int for fully
            reproducible standard errors; None (default) draws fresh entropy.

        Attributes set
        --------------
        threshold_se : dict
            {item: numpy.ndarray} bootstrap SEs for each item's uncentred thresholds.
        item_se : pandas.Series
            Bootstrap SE for each item's central location.
        cat_width_se : dict
            {item: numpy.ndarray} bootstrap SEs for category widths.
        threshold_low / threshold_high : dict or None
            Bootstrap CI bounds for thresholds, or None.
        central_bootstrap : pandas.DataFrame
            Bootstrap central item location estimates, shape (no_of_samples, items).
        threshold_bootstrap : dict
            {item: DataFrame} of bootstrap threshold estimates.
        """
        rng = np.random.default_rng(seed)
        samples = [
            PCM(self.responses.sample(frac=1, replace=True, random_state=rng), self.max_score_vector)
            for _ in range(no_of_samples)
        ]

        for sample in samples:
            sample.calibrate(
                constant=constant,
                robust=robust,
                method=method,
                matrix_power=matrix_power,
                log_lik_tol=log_lik_tol,
            )

        self.bootstrap_sample_thresholds = {
            f"Sample_{i + 1}": sample.thresholds_uncentred
            for i, sample in enumerate(samples)
        }

        calibrations_thresholds = {
            item: np.stack(
                [
                    samples[s].thresholds_uncentred.loc[item].dropna().values
                    for s in range(no_of_samples)
                ]
            )
            for item in self.item_names
        }
        calibrations_central = {
            item: np.array([samples[s].items[item] for s in range(no_of_samples)])
            for item in self.item_names
        }

        sample_index = [f"Sample {i + 1}" for i in range(no_of_samples)]

        self.central_bootstrap = pd.DataFrame(calibrations_central, index=sample_index)

        self.threshold_bootstrap = {}
        for item in self.item_names:
            df_b = pd.DataFrame(calibrations_thresholds[item], index=sample_index)
            df_b.columns = np.arange(1, df_b.shape[1] + 1)
            self.threshold_bootstrap[item] = df_b

        self.cat_width_bootstrap = {}
        for item in self.item_names:
            if self.max_score_vector[item] == 1:
                self.cat_width_bootstrap[item] = pd.DataFrame(
                    {1: pd.Series({f"Sample {i + 1}": 0 for i in range(no_of_samples)})}
                )
            else:
                tb = self.threshold_bootstrap[item]
                cwb = pd.DataFrame(index=sample_index)
                for score in range(self.max_score_vector[item] - 1):
                    cwb[score + 1] = tb[score + 2] - tb[score + 1]
                self.cat_width_bootstrap[item] = cwb

        self.threshold_se = {
            item: np.std(calibrations_thresholds[item], axis=0)
            for item in self.responses.columns
        }
        self.cat_width_se = {
            item: np.std(self.cat_width_bootstrap[item], axis=0)
            for item in self.item_names
        }
        self.item_se = pd.Series(
            {
                item: np.std(calibrations_central[item])
                for item in self.responses.columns
            }
        )

        if interval is not None:
            lo = (1 - interval) * 50
            hi = (1 + interval) * 50
            self.threshold_low = {
                item: np.percentile(calibrations_thresholds[item], lo, axis=0)
                for item in self.responses.columns
            }
            self.threshold_high = {
                item: np.percentile(calibrations_thresholds[item], hi, axis=0)
                for item in self.responses.columns
            }
            self.cat_width_low = {
                item: np.percentile(self.cat_width_bootstrap[item], lo, axis=0)
                for item in self.responses.columns
            }
            self.cat_width_high = {
                item: np.percentile(self.cat_width_bootstrap[item], hi, axis=0)
                for item in self.responses.columns
            }
            self.central_low = pd.Series(
                {
                    item: np.percentile(calibrations_central[item], lo)
                    for item in self.responses.columns
                }
            )
            self.central_high = pd.Series(
                {
                    item: np.percentile(calibrations_central[item], hi)
                    for item in self.responses.columns
                }
            )
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
            Person location estimates in logits. Returns numpy.nan on failure.
        """
        if isinstance(persons, str):
            persons = self.person_names if persons == "all" else [persons]

        if items is None:
            items = list(self.item_names)
        elif isinstance(items, str):
            items = list(self.item_names) if items in ("all",) else [items]

        thresholds = self.thresholds_uncentred
        person_data = self.responses.loc[persons, items]

        if missing_as_incorrect:
            person_data = person_data.fillna(0)

        person_filter = person_data.notna().astype(float)

        scores = person_data.sum(axis=1).astype(float)
        ext_scores = person_filter.mul(self.max_score_vector[items], axis=1).sum(axis=1)

        scores = scores.clip(lower=ext_score_adjustment)
        scores[scores == ext_scores] -= ext_score_adjustment
        # Avoid modifying original if ext_scores match before clip adjustment
        scores = scores.where(scores < ext_scores, ext_scores - ext_score_adjustment)

        thresh_sums = pd.Series(
            {item: thresholds.loc[item].dropna().sum() for item in items}
        )
        # Weighted mean threshold per person (accounting for missing items)
        thresh_sum_df = person_filter.mul(thresh_sums, axis=1)
        max_score_df = person_filter.mul(self.max_score_vector[items], axis=1)
        mean_item_locations = thresh_sum_df.sum(axis=1) / max_score_df.sum(axis=1)

        try:
            estimates = np.log(scores) - np.log(ext_scores - scores) + mean_item_locations
            items_list = list(items)

            # Per-person convergence mask. The original code accidentally used
            # nan propagation from exp() overflow to freeze converged persons
            # (abs(nan) > tol = False). Our log-sum-exp implementation gives
            # valid probabilities for all person location values, so we must track
            # convergence explicitly: once a person's change drops below
            # tolerance, exclude them from further updates. Without this,
            # slowly-converging persons keep updating everyone, and persons
            # near extreme scores accumulate drift of +/-1 logit per iteration
            # over max_iters steps, producing e.g. person_location=117 logits.
            active = pd.Series(True, index=persons)
            iters = 0

            while active.any() and iters <= max_iters:
                active_idx = active[active].index

                probs, cats_arr = self._cat_probs_matrix(
                    estimates.loc[active_idx].values, items_list, thresholds
                )
                # probs: (C, N_active, I)
                exp_score = (cats_arr[:, None, None] * probs).sum(
                    axis=0
                )  # (N_active, I)
                exp_score_df = pd.DataFrame(
                    exp_score, index=active_idx, columns=items_list
                )
                exp_score_df *= person_filter.loc[active_idx]

                dev = (
                    cats_arr[:, None, None] - exp_score[None, :, :]
                )  # (C, N_active, I)
                info = (dev**2 * probs).sum(axis=0)  # (N_active, I)
                info_df = pd.DataFrame(info, index=active_idx, columns=items_list)
                info_df *= person_filter.loc[active_idx]

                result_list = exp_score_df.sum(axis=1)
                info_list = info_df.sum(axis=1)

                changes = ((result_list - scores.loc[active_idx]) / info_list).clip(
                    -1, 1
                )
                estimates.loc[active_idx] -= changes

                # Freeze persons whose change is now within tolerance
                active.loc[active_idx] = abs(changes) > tolerance
                iters += 1

            if iters >= max_iters and active.any():
                n_nc = int(active.sum())
                warnings.warn(
                    f"{n_nc} person(s) did not converge in person() and will be set to NaN. "
                    f"Consider increasing max_iters or checking for degenerate response patterns.",
                    UserWarning,
                    stacklevel=2,
                )
                estimates[active] = np.nan

            if warm_corr:
                # Apply Warm correction only to persons with valid (finite) estimates
                valid = estimates.notna()
                if valid.any():
                    estimates[valid] += self.warm(
                        estimates[valid],
                        items_list,
                        person_filter.loc[estimates.index[valid]],
                    )

        except Exception as e:
            warnings.warn(
                f"person() failed with exception: {e}. "
                "Returning NaN for all persons.",
                UserWarning,
                stacklevel=2,
            )
            estimates = pd.Series(np.nan, index=persons)

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

        Wrapper around person() that estimates person locations for all persons
        and stores the result as self.persons.

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
            Person location estimate in logits.
        """
        # BUG FIX: original had a string-iteration bug when items was a single string item name.
        if items is None or (isinstance(items, str) and items in ("all", "none")):
            items = list(self.item_names)
        elif isinstance(items, str):
            items = [items]

        thresholds = self.thresholds_uncentred
        mean_item_location = thresholds.stack().mean()
        ext_score = self.max_score_vector[items].sum()

        used_score = float(score)
        if used_score == 0:
            used_score = ext_score_adjustment
        elif used_score == ext_score:
            used_score -= ext_score_adjustment

        estimate = log(used_score) - log(ext_score - used_score) + mean_item_location
        change = 1.0
        iters = 0

        while abs(change) > tolerance and iters <= max_iters:
            result = sum(
                self.exp_score_uncentred(estimate, thresholds.loc[item].dropna())
                for item in items
            )
            info = sum(
                self.variance_uncentred(estimate, thresholds.loc[item].dropna())
                for item in items
            )
            change = max(-1.0, min(1.0, (result - used_score) / info))
            estimate -= change
            iters += 1

        if warm_corr:
            person_filter = {item: True for item in items}
            estimate += self.warm(
                pd.Series({"score": estimate}),
                items,
                pd.DataFrame(person_filter, index=["score"]),
            ).iloc[0]

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

        Estimates the person location corresponding to every possible raw score on
        a given item set using vectorised Newton-Raphson, and stores the
        result as self.score_table.

        Parameters
        ----------
        items : str, list, or None, default None
            Item subset to use. None or 'all' uses all items.
        ext_scores : bool, default True
            If True, includes extreme scores (0 and maximum) in the table,
            adjusted by ext_score_adjustment.
        warm_corr : bool, default True
            If True, applies Warm's (1989) bias correction.
        tolerance : float, default 0.00001
            Newton-Raphson convergence tolerance.
        max_iters : int, default 100
            Maximum Newton-Raphson iterations.
        ext_score_adjustment : float, default 0.5
            Adjustment applied to extreme scores when ext_scores=True.

        Attributes set
        --------------
        score_table : pandas.Series
            Person location estimate for each possible raw score, indexed by score.

        Returns
        -------
        None
        """

        if isinstance(items, str) and items in ("all", "none"):
            items = None
        elif isinstance(items, str):
            items = [items]
        if items is None:
            items = list(self.item_names)

        thresholds = self.thresholds_uncentred
        total_max = self.max_score_vector.loc[items].sum()

        if ext_scores:
            scores = np.arange(total_max + 1)
            used_scores = scores.astype(float)
            used_scores[0] += ext_score_adjustment
            used_scores[-1] -= ext_score_adjustment
        else:
            scores = np.arange(1, total_max)
            used_scores = scores.astype(float)

        mean_item_location = thresholds.stack().mean()
        estimates = pd.Series(
            np.log(used_scores) - np.log(total_max - used_scores) + mean_item_location,
            index=scores,
        )

        changes = pd.Series(1.0, index=scores)
        iters = 0

        while abs(changes).max() > tolerance and iters <= max_iters:
            probs, cats_arr = self._cat_probs_matrix(
                estimates.values, items, thresholds
            )
            exp_score = (cats_arr[:, None, None] * probs).sum(axis=0)  # (N, I)
            exp_score_df = pd.DataFrame(exp_score, index=scores, columns=items)

            dev = cats_arr[:, None, None] - exp_score[None, :, :]
            info = (dev**2 * probs).sum(axis=0)
            info_df = pd.DataFrame(info, index=scores, columns=items)

            result_list = exp_score_df.sum(axis=1)
            info_list = info_df.sum(axis=1)

            changes = ((result_list - used_scores) / info_list).clip(-1, 1)
            estimates -= changes
            iters += 1

        if warm_corr:
            person_filter = pd.DataFrame(1.0, columns=items, index=scores)
            estimates += self.warm(estimates, items, person_filter)

        self.score_table = estimates

    def warm(self, person_locations, items, person_filter):
        """
        Apply Warm's (1989) weighted maximum likelihood bias correction.

        Uses the vectorised _cat_probs_matrix engine. Computes
        (J1 - J2 + J3) / (2 * I^2) simultaneously for all persons.

        Parameters
        ----------
        person_locations : pandas.Series
            Current person location estimates.
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
        thresholds = self.thresholds_uncentred

        probs, cats_arr = self._cat_probs_matrix(person_locations.values, items, thresholds)
        # probs: (C, N, I)

        pf = person_filter.values if isinstance(person_filter, pd.DataFrame) else None

        exp_score = (cats_arr[:, None, None] * probs).sum(axis=0)  # (N, I)
        if pf is not None:
            exp_score *= pf

        dev = cats_arr[:, None, None] - exp_score[None, :, :]  # (C, N, I)
        info = (dev**2 * probs).sum(axis=0)  # (N, I)
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
        # of the correct -0.63 -- the source of person location estimates of +269 logits.
        cats3 = (cats_arr**3)[:, None, None]
        masked_probs = probs * pf[None, :, :] if pf is not None else probs  # (C, N, I)
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

        CSEM = 1 / sqrt(I) where I is total Fisher information.

        Parameters
        ----------
        persons : list, str, or None, default None
            Person identifiers. Overrides person_locations if provided.
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
            CSEM values in logits.
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

        persons = person_locations.index

        # BUG FIX: original `else` branch set thresholds = self.thresholds_uncentred.loc[items].dropna()
        # which returns a column slice (wrong type) for a list of items.
        if items is None or (isinstance(items, str) and items == "all"):
            items = list(self.item_names)
        elif isinstance(items, str):
            items = [items]

        thresholds = self.thresholds_uncentred

        # BUG FIX (original): unconditionally indexed self.responses by persons,
        # which failed for hypothetical locations (raw floats/lists, or any
        # person_locations index not matching self.responses) with no matching
        # row. Real persons are still filtered by their actual missing-response
        # pattern; hypothetical locations are treated as fully answered.
        is_real_person = persons.isin(self.responses.index)
        person_filter = self.responses.reindex(persons)[items].notna().astype(float)
        person_filter.loc[~is_real_person] = 1.0

        probs, cats_arr = self._cat_probs_matrix(person_locations.values, items, thresholds)
        exp_score = (cats_arr[:, None, None] * probs).sum(axis=0)
        exp_score_df = pd.DataFrame(exp_score, index=persons, columns=items)
        exp_score_df *= person_filter

        dev = cats_arr[:, None, None] - exp_score[None, :, :]
        info = (dev**2 * probs).sum(axis=0)
        info_df = pd.DataFrame(info, index=persons, columns=items)
        info_df *= person_filter

        return 1.0 / (info_df.sum(axis=1) ** 0.5)

    # ------------------------------------------------------------------
    # Descriptive / count methods
    # ------------------------------------------------------------------

    def category_counts_df(self, persons=None, items=None, counts_name=None):
        """
        Build a response frequency table for one or more persons, across
        one or more items.

        Computes per-item response counts for each valid category, total
        valid responses, and missing responses, over the requested
        persons. Items with different maximum scores show blank cells
        (not 0) for categories above their maximum, to avoid implying
        those categories exist. Appends a 'Total' row.

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
            DataFrame with items as rows and response categories (0 to
            max max_score), 'Total', and 'Missing' as columns. Cells
            above an item's max_score are blank. A 'Total' row is
            appended.
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

        # Build per-item counts reindexed to that item's valid score range only.
        # Items with different max scores will have NaN for categories above their
        # maximum when combined into a single DataFrame -- these should display as
        # blank cells, not 0, to avoid implying those categories exist for that item.
        cat_counts_dict = {
            item: subset[item]
            .value_counts()
            .reindex(range(self.max_score_vector[item] + 1), fill_value=0)
            .astype(int)
            for item in items
        }
        category_counts_df = pd.DataFrame(cat_counts_dict).T
        category_counts_df.sort_index(axis=1, inplace=True)

        category_counts_df["Total"] = subset.count()
        category_counts_df["Missing"] = len(persons) - category_counts_df["Total"]
        category_counts_df.loc["Total"] = category_counts_df.sum()

        # Convert valid counts to int, then replace NaN (categories above item
        # max score) with '' so the output table shows a blank rather than 0.
        category_counts_df = category_counts_df.fillna(-1).astype(int).replace(-1, "")

        if counts_name is None:
            self.category_counts_table = category_counts_df
        else:
            if not hasattr(self, "counts"):
                from raschpy.base import _Namespace

                self.counts = _Namespace()
            self.counts[counts_name] = category_counts_df

        return category_counts_df

    # ------------------------------------------------------------------
    # Fit statistics
    # ------------------------------------------------------------------

    def _log_likelihood(self, responses=None, persons=None):
        if responses is None:
            responses = self.responses
        if persons is None:
            persons = self.persons
        scores = responses.sum(axis=1)
        max_scores = responses.notna().mul(self.max_score_vector, axis=1).sum(axis=1)
        non_extreme = responses.index[(scores > 0) & (scores < max_scores)]
        persons = persons.reindex(non_extreme).dropna()
        persons = persons[persons.abs() <= 20]
        obs_arr = responses.loc[persons.index].values
        valid = ~np.isnan(obs_arr)
        probs, _ = self._cat_probs_matrix(
            persons.values, list(self.item_names), self.thresholds_uncentred
        )
        obs_int = np.where(valid, obs_arr, 0).astype(int)
        n_idx, i_idx = np.meshgrid(
            np.arange(obs_arr.shape[0]), np.arange(obs_arr.shape[1]), indexing="ij"
        )
        prob_obs = probs[obs_int, n_idx, i_idx]
        prob_obs[~valid] = np.nan
        return float(np.nansum(np.log(prob_obs)))

    def _item_location_log_likelihood(
        self, responses, theta, step_shape, item_names,
        tolerance=0.00001, max_iters=100,
    ):
        """
        Log-likelihood maximised over a per-item location shift only, holding
        person location (theta) and the per-item category step-shape both fixed.

        Used by dif_test(omnibus_scope='item') to isolate item-location DIF
        from threshold/category-structure DIF: PCM has a separate threshold
        vector per item (unlike SLM, which has none, or RSM, which shares one
        vector across all items), so simply reusing each group's own
        independently-calibrated thresholds_uncentred for the omnibus H1 side
        leaves items*(max_score-1) threshold parameters free to differ
        between groups without being charged to df=items-1 — this is the
        dominant source of the false-positive inflation flagged by the user
        (a much smaller residual than the same issue for SLM/RSM, which have
        far fewer or no per-item threshold parameters). Fixing step_shape
        (the pooled model's own zero-centred per-item category structure)
        and theta (pooled person locations), then only re-optimising a single
        location shift per item via Newton-Raphson, correctly restricts H1
        vs H0's free-parameter difference to exactly items-1, mirroring how
        pooled person locations alone were sufficient for SLM/RSM/MFRM.

        Newton-Raphson update mirrors person()'s location-estimation loop
        (same expected-score/information-based step), but the sign is
        flipped and the sum is over persons per item rather than over items
        per person, since increasing an item's location shift decreases
        expected score (the opposite direction to increasing person location).

        Parameters
        ----------
        responses : pandas.DataFrame
            This group's own response data (already restricted to its own
            persons).
        theta : pandas.Series
            Fixed person location estimates (e.g. from the pooled model), indexed by
            person.
        step_shape : pandas.DataFrame
            Fixed per-item category step-shape (pooled model's
            thresholds_uncentred minus its own item locations), indexed
            by item, zero-centred per item.
        item_names : list of str
            Items to include (the common item set between groups).

        Returns
        -------
        float
            Log-likelihood at the per-item-optimised location shifts.
        """
        scores = responses.sum(axis=1)
        max_scores = responses.notna().mul(self.max_score_vector, axis=1).sum(axis=1)
        non_extreme = responses.index[(scores > 0) & (scores < max_scores)]
        theta = theta.reindex(non_extreme).dropna()
        theta = theta[theta.abs() <= 20]
        resp = responses.loc[theta.index, item_names]
        person_filter = resp.notna().astype(float)
        obs_scores = resp.fillna(0).sum(axis=0)

        delta = pd.Series(0.0, index=item_names)
        active = pd.Series(True, index=item_names)
        iters = 0
        while active.any() and iters <= max_iters:
            active_items = active[active].index.tolist()
            thresholds_now = step_shape.add(delta, axis=0)
            probs, cats_arr = self._cat_probs_matrix(
                theta.values, active_items, thresholds_now.loc[active_items]
            )
            exp_score = (cats_arr[:, None, None] * probs).sum(axis=0)  # (N, I_active)
            exp_score_df = pd.DataFrame(exp_score, index=theta.index, columns=active_items)
            exp_score_df *= person_filter[active_items]
            dev = cats_arr[:, None, None] - exp_score[None, :, :]
            info = (dev ** 2 * probs).sum(axis=0)
            info_df = pd.DataFrame(info, index=theta.index, columns=active_items)
            info_df *= person_filter[active_items]

            exp_sum = exp_score_df.sum(axis=0)
            info_sum = info_df.sum(axis=0)

            changes = ((exp_sum - obs_scores[active_items]) / info_sum).clip(-1, 1)
            delta.loc[active_items] += changes
            active.loc[active_items] = changes.abs() > tolerance
            iters += 1

        thresholds_final = step_shape.add(delta, axis=0)
        probs, cats_arr = self._cat_probs_matrix(theta.values, item_names, thresholds_final)
        obs_arr = resp.values
        valid = ~np.isnan(obs_arr)
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
        constant=None,
        method="log-lik",
        robust=False,
        matrix_power=None,
        log_lik_tol=0.000001,
        no_of_samples=500,
        interval=None,
        seed=None,
    ):
        """
        Compute all item, threshold, person, and test-level fit statistics.

        The central computation method. Auto-triggers calibrate(), std_errors(),
        and person_estimates() if not yet run. Computes expected scores,
        information, kurtosis, residuals, and standardised residuals for all
        person-item combinations, then derives Infit and Outfit mean-square
        and Z statistics (Wilson-Hilferty approximation) at both item and
        threshold level. Optionally computes test-level separation statistics.

        Parameters
        ----------
        warm_corr : bool, default True
            If True, applies Warm's (1989) bias correction to person location
            estimates used in fit computation.
        se : bool, default True
            If True, computes bootstrap standard errors. If False,
            test_stats is forced False.
        test_stats : bool, default True
            If True, computes test-level statistics (ISI, PSI, strata,
            reliability). Requires se=True.
        trim_cat_prob_dict : bool, default False
            If True, restricts cat_prob_dict to non-extreme persons.
            Reduces memory usage on large datasets.
        tolerance : float, default 0.00001
            Newton-Raphson convergence tolerance for person location estimation.
        max_iters : int, default 100
            Maximum Newton-Raphson iterations.
        ext_score_adjustment : float, default 0.5
            Extreme score adjustment for person location estimation.
        constant : float, default 0.1
            Additive smoothing constant for calibration.
        method : str, default 'log-lik'
            Priority vector extraction method for calibration.
        matrix_power : int or None, default None
            Matrix power for calibration. None resolves to 0 for
            method='log-lik' (no powering), else 5.
        log_lik_tol : float, default 0.000001
            Log-likelihood tolerance for calibration.
        no_of_samples : int, default 500
            Bootstrap samples for standard error estimation.
        interval : float or None, default None
            Confidence interval width for bootstrap CIs.
        seed : int or None, default None
            Seed passed through to the internal std_errors() call (only
            used if item SEs aren't already computed). None draws fresh
            entropy each call.

        Attributes set
        --------------
        cat_prob_dict : dict
            {cat: DataFrame} of category probabilities for all persons and items.
        exp_score_df : pandas.DataFrame
            Expected scores (non-extreme persons only), shape (persons, items).
        info_df : pandas.DataFrame
            Fisher information values, shape (persons, items).
        kurtosis_df : pandas.DataFrame
            Fourth central moments, shape (persons, items).
        residual_df : pandas.DataFrame
            Raw residuals (observed - expected), shape (persons, items).
        std_residual_df : pandas.DataFrame
            Standardised residuals, shape (persons, items).
        item_infit_ms : pandas.Series
            Item infit mean-square statistics.
        item_outfit_ms : pandas.Series
            Item outfit mean-square statistics.
        item_infit_zstd : pandas.Series
            Item infit Z statistics (Wilson-Hilferty approximation).
        item_outfit_zstd : pandas.Series
            Item outfit Z statistics.
        threshold_infit_ms : pandas.Series
            Threshold-level infit MS (MultiIndex: item, threshold).
        threshold_outfit_ms : pandas.Series
            Threshold-level outfit MS.
        threshold_infit_zstd : pandas.Series
            Threshold-level infit Z.
        threshold_outfit_zstd : pandas.Series
            Threshold-level outfit Z.
        response_counts : pandas.Series
            Number of valid responses per item.
        item_facilities : pandas.Series
            Mean response / max_score per item.
        point_measure : pandas.Series
            Observed point-measure correlations per item.
        exp_point_measure : pandas.Series
            Expected point-measure correlations per item.
        csem_vector : pandas.Series
            Conditional SEM for each person.
        rsem_vector : pandas.Series
            Residual SEM for each person.
        person_infit_ms : pandas.Series
            Person infit mean-square statistics.
        person_outfit_ms : pandas.Series
            Person outfit mean-square statistics.
        person_infit_zstd : pandas.Series
            Person infit Z statistics.
        person_outfit_zstd : pandas.Series
            Person outfit Z statistics.
        isi_central : float
            Item separation index based on central item locations (if test_stats).
        isi_thresholds : float
            Item separation index based on thresholds (if test_stats).
        item_strata : float
            Number of statistically distinct item strata (if test_stats).
        item_reliability : float
            Item reliability coefficient (if test_stats).
        psi : float
            Person separation index (if test_stats).
        person_strata : float
            Number of statistically distinct person strata (if test_stats).
        person_reliability : float
            Person reliability coefficient (if test_stats).

        Returns
        -------
        None
        """

        if not hasattr(self, "thresholds_uncentred"):
            self.calibrate(constant=constant, method=method, robust=robust)
        if se and not hasattr(self, "threshold_se"):
            self.std_errors(
                interval=interval,
                no_of_samples=no_of_samples,
                constant=constant,
                robust=robust,
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

        df = self.responses.copy()
        scores = df.sum(axis=1)
        max_scores = df.notna().mul(self.max_score_vector, axis=1).sum(axis=1)
        df = df[(scores > 0) & (scores < max_scores)]
        missing_mask = df.notna().astype(float)
        person_locations = self.persons.loc[df.index]

        # Safety net: exclude persons with extreme person location estimates.
        # Diverged NR iterations (e.g. near-perfect scorers on sparse response
        # patterns) can produce finite but astronomically large estimates such as
        # +117 logits. These are set to NaN in person() when non-convergence is
        # detected, but guard here as well against any that slip through.
        # |person location| > 20 logits is well beyond any plausible true parameter value
        # and would produce kurtosis/info^2 ~ 1e+60 in the outfit q-factor.
        person_locations = person_locations[person_locations.abs() <= 20]
        df = df.loc[person_locations.index]
        missing_mask = missing_mask.loc[person_locations.index]

        item_count = df.notna().sum(axis=0)
        person_count = df.notna().sum(axis=1)

        items_list = list(self.item_names)
        probs, cats_arr = self._cat_probs_matrix(
            person_locations.values, items_list, self.thresholds_uncentred
        )
        # probs: (C, N, I)

        # Store cat_prob_dict for downstream use (e.g. trim_cat_prob_dict)
        self.cat_prob_dict = {
            cat: pd.DataFrame(
                probs[cat], index=person_locations.index, columns=self.item_names
            )
            for cat in range(probs.shape[0])
        }
        if trim_cat_prob_dict:
            for cat in self.cat_prob_dict:
                self.cat_prob_dict[cat] = self.cat_prob_dict[cat].loc[df.index]

        self.log_likelihood = self._log_likelihood()

        n_persons = int(((scores > 0) & (scores < max_scores)).sum())
        k = int(self.thresholds_uncentred.notna().sum().sum()) - 1
        self.aic = 2 * k - 2 * self.log_likelihood
        self.bic = k * np.log(n_persons) - 2 * self.log_likelihood

        exp_score = (cats_arr[:, None, None] * probs).sum(axis=0)  # (N, I)
        self.exp_score_df = pd.DataFrame(
            exp_score, index=person_locations.index, columns=self.item_names
        )
        self.exp_score_df *= missing_mask

        dev = cats_arr[:, None, None] - exp_score[None, :, :]  # (C, N, I)
        info = (dev**2 * probs).sum(axis=0)  # (N, I)
        self.info_df = pd.DataFrame(
            info, index=person_locations.index, columns=self.item_names
        )
        self.info_df *= missing_mask

        kurtosis = ((dev**4) * probs).sum(axis=0)  # (N, I)
        self.kurtosis_df = pd.DataFrame(
            kurtosis, index=person_locations.index, columns=self.item_names
        )
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
            probs.max(axis=0),  # (N, I): max prob across categories
            index=person_locations.index,
            columns=self.item_names,
        )
        degenerate_mask = max_cat_prob > P_THRESHOLD
        self.info_df[degenerate_mask] = np.nan
        self.kurtosis_df[degenerate_mask] = np.nan
        self.exp_score_df[degenerate_mask] = np.nan

        self.residual_df = self.responses.reindex(df.index) - self.exp_score_df
        self.std_residual_df = self.residual_df / (self.info_df**0.5)

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
        self.item_facilities = self.responses.mean(axis=0) / self.max_score_vector

        (self.point_measure, self.exp_point_measure) = self.pt_meas(
            self.persons, self.exp_score_df, self.info_df
        )

        # --- Threshold fit (dichotomised) ---
        dich_thresh = {}
        for item in self.responses.columns:
            dich_thresh[item] = {}
            for t in range(self.max_score_vector[item]):
                col = (
                    self.responses[item].where(
                        self.responses[item].isin([t, t + 1]), np.nan
                    )
                    - t
                )
                dich_thresh[item][t + 1] = col

        dich_thresh_exp = {item: {} for item in self.responses.columns}
        dich_thresh_var = {item: {} for item in self.responses.columns}
        dich_thresh_kur = {item: {} for item in self.responses.columns}
        dich_residuals = {item: {} for item in self.responses.columns}
        dich_std_residuals = {item: {} for item in self.responses.columns}

        dich_thresh_count = {
            item: {
                t + 1: dich_thresh[item][t + 1].count()
                for t in range(self.max_score_vector[item])
            }
            for item in self.responses.columns
        }

        for item in self.responses.columns:
            thresh_val = self.thresholds_uncentred.loc[item].dropna()
            for t in range(self.max_score_vector[item]):
                diff = thresh_val.iloc[t]
                mm = dich_thresh[item][t + 1].notna().astype(float).replace(0, np.nan)

                p = 1.0 / (1.0 + np.exp(diff - self.persons))
                p_masked = p * mm

                dich_thresh_exp[item][t + 1] = p_masked
                var = p_masked * (1 - p_masked)
                dich_thresh_var[item][t + 1] = var

                # Kurtosis for binary item
                dich_thresh_kur[item][t + 1] = (
                    ((-p_masked) ** 4) * (1 - p_masked)
                    + ((1 - p_masked) ** 4) * p_masked
                ) * mm

                dich_residuals[item][t + 1] = dich_thresh[item][t + 1] - p_masked
                dich_std_residuals[item][t + 1] = dich_residuals[item][t + 1] / (
                    var**0.5
                )

        def _concat_series(nested_dict):
            """Helper: flatten {item: {t: Series}} -> MultiIndex Series."""
            return pd.concat(
                {item: pd.Series(nested_dict[item]) for item in self.responses.columns},
                keys=self.responses.columns,
            )

        self.threshold_outfit_ms = _concat_series(
            {
                item: {
                    t
                    + 1: (
                        (dich_std_residuals[item][t + 1] ** 2).sum()
                        / dich_thresh_count[item][t + 1]
                        if dich_thresh_count[item][t + 1] > 0
                        else np.nan
                    )
                    for t in range(self.max_score_vector[item])
                }
                for item in self.responses.columns
            }
        )

        self.threshold_infit_ms = _concat_series(
            {
                item: {
                    t
                    + 1: (
                        (dich_residuals[item][t + 1] ** 2).sum()
                        / dich_thresh_var[item][t + 1].sum()
                        if dich_thresh_var[item][t + 1].sum() > 0
                        else np.nan
                    )
                    for t in range(self.max_score_vector[item])
                }
                for item in self.responses.columns
            }
        )

        threshold_outfit_q_raw = _concat_series(
            {
                item: {
                    t
                    + 1: (
                        (
                            (
                                dich_thresh_kur[item][t + 1]
                                / (dich_thresh_var[item][t + 1] ** 2)
                            )
                            / (dich_thresh_count[item][t + 1] ** 2)
                        ).sum()
                        - (1 / dich_thresh_count[item][t + 1])
                        if dich_thresh_count[item][t + 1] > 0
                        else np.nan
                    )
                    for t in range(self.max_score_vector[item])
                }
                for item in self.responses.columns
            }
        )
        threshold_outfit_q = threshold_outfit_q_raw**0.5
        self.threshold_outfit_zstd = ((self.threshold_outfit_ms ** (1 / 3)) - 1) * (
            3 / threshold_outfit_q
        ) + (threshold_outfit_q / 3)

        threshold_infit_q = (
            _concat_series(
                {
                    item: {
                        t
                        + 1: (
                            (
                                dich_thresh_kur[item][t + 1]
                                - dich_thresh_var[item][t + 1] ** 2
                            ).sum()
                            / (dich_thresh_var[item][t + 1].sum() ** 2)
                            if dich_thresh_var[item][t + 1].sum() > 0
                            else np.nan
                        )
                        for t in range(self.max_score_vector[item])
                    }
                    for item in self.responses.columns
                }
            )
            ** 0.5
        )
        self.threshold_infit_zstd = ((self.threshold_infit_ms ** (1 / 3)) - 1) * (
            3 / threshold_infit_q
        ) + (threshold_infit_q / 3)

        person_location_deviation = self.persons - self.persons.mean()

        # Threshold point-measure correlations
        pm_num = _concat_series(
            {
                item: {
                    t
                    + 1: (
                        (dich_thresh[item][t + 1] - dich_thresh[item][t + 1].mean())
                        * person_location_deviation
                    ).sum()
                    for t in range(self.max_score_vector[item])
                }
                for item in self.responses.columns
            }
        )
        pm_den = _concat_series(
            {
                item: {
                    t
                    + 1: (
                        (
                            (dich_thresh[item][t + 1] - dich_thresh[item][t + 1].mean())
                            ** 2
                        ).sum()
                        * (person_location_deviation**2).sum()
                    )
                    ** 0.5
                    for t in range(self.max_score_vector[item])
                }
                for item in self.responses.columns
            }
        )
        self.threshold_point_measure = pm_num / pm_den

        exp_pm_dict = {
            item: {
                t
                + 1: (
                    dich_thresh_exp[item][t + 1] - dich_thresh_exp[item][t + 1].mean()
                    if dich_thresh_exp[item][t + 1].count() > 0
                    else np.nan
                )
                for t in range(self.max_score_vector[item])
            }
            for item in self.responses.columns
        }
        exp_pm_num = _concat_series(
            {
                item: {
                    t + 1: (exp_pm_dict[item][t + 1] * person_location_deviation).sum()
                    for t in range(self.max_score_vector[item])
                }
                for item in self.responses.columns
            }
        )
        exp_pm_den_raw = _concat_series(
            {
                item: {
                    t
                    + 1: (
                        (exp_pm_dict[item][t + 1] ** 2) + dich_thresh_var[item][t + 1]
                    ).sum()
                    for t in range(self.max_score_vector[item])
                }
                for item in self.responses.columns
            }
        )
        exp_pm_den = (exp_pm_den_raw * (person_location_deviation**2).sum()) ** 0.5
        self.threshold_exp_point_measure = exp_pm_num / exp_pm_den

        self.threshold_rmsr = (
            _concat_series(
                {
                    item: {
                        t
                        + 1: (
                            (dich_residuals[item][t + 1] ** 2).sum()
                            / dich_residuals[item][t + 1].count()
                            if dich_residuals[item][t + 1].count() > 0
                            else np.nan
                        )
                        for t in range(self.max_score_vector[item])
                    }
                    for item in self.responses.columns
                }
            )
            ** 0.5
        )

        diff_num = _concat_series(
            {
                item: {
                    t
                    + 1: (
                        (
                            self.persons
                            - self.thresholds_uncentred.loc[item].dropna().iloc[t]
                        )
                        * dich_residuals[item][t + 1]
                    ).sum()
                    for t in range(self.max_score_vector[item])
                }
                for item in self.responses.columns
            }
        )
        diff_den = _concat_series(
            {
                item: {
                    t
                    + 1: (
                        dich_thresh_var[item][t + 1]
                        * (
                            self.persons
                            - self.thresholds_uncentred.loc[item].dropna().iloc[t]
                        )
                        ** 2
                    ).sum()
                    for t in range(self.max_score_vector[item])
                }
                for item in self.responses.columns
            }
        )
        self.threshold_discrimination = 1 + diff_num / diff_den

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
            thresh_flat = np.concatenate(
                [
                    self.thresholds_uncentred.loc[item].dropna().values
                    for item in self.responses.columns
                ]
            )
            se_flat = np.concatenate(
                [self.threshold_se[item] for item in self.responses.columns]
            )
            self.threshold_list = thresh_flat
            self.threshold_se_list = se_flat

            self.isi_central = (self.items.var() / (self.item_se**2).mean() - 1) ** 0.5
            self.item_strata = (4 * self.isi_central + 1) / 3
            self.item_reliability = self.isi_central**2 / (1 + self.isi_central**2)

            self.isi_thresholds = (thresh_flat.var() / (se_flat**2).mean() - 1) ** 0.5
            self.threshold_strata = (4 * self.isi_thresholds + 1) / 3
            self.threshold_reliability = self.isi_thresholds**2 / (
                1 + self.isi_thresholds**2
            )

            self.psi = (
                np.var(self.persons) - (self.rsem_vector**2).mean()
            ) ** 0.5 / (self.rsem_vector**2).mean() ** 0.5
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
        method="log-lik",
        robust=False,
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
        constant, method, log_lik_tol : floats
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
            Degrees of freedom (total thresholds across all items - 1).
        andersen_p : float
            p-value from chi-squared distribution.
        andersen_groups : dict
            {group_name: PCM} — fitted group models for inspection. Group
            names are 'low'/'high' for split_by='person_location'/'score', or the
            two observed covariate values for split_by='exogenous'.
        andersen_summary : pandas.Series
            LR statistic, df, and p-value.
        """
        from raschpy.pcm import PCM

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

        if not hasattr(self, "thresholds_uncentred"):
            self.calibrate(constant=constant, method=method, robust=robust, log_lik_tol=log_lik_tol)
        if not hasattr(self, "persons"):
            self.person_estimates(
                warm_corr=warm_corr,
                tolerance=tolerance,
                max_iters=max_iters,
                ext_score_adjustment=ext_score_adjustment,
            )

        scores = self.responses.sum(axis=1)
        max_scores = self.responses.notna().mul(self.max_score_vector, axis=1).sum(axis=1)
        non_extreme = self.responses.index[(scores > 0) & (scores < max_scores)]

        group_idx = self._resolve_andersen_groups(
            split_by, covariate, non_extreme, self.persons, scores
        )

        # Full-model LL restricted to persons in either group
        combined_idx = group_idx[list(group_idx)[0]].append(group_idx[list(group_idx)[1]])

        # Re-estimate full model on combined subset so the LR comparison is fair
        m_full = PCM(self.responses.loc[combined_idx], self.max_score_vector)
        m_full.calibrate(constant=constant, method=method, robust=robust, log_lik_tol=log_lik_tol)
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
            m = PCM(self.responses.loc[idx], self.max_score_vector)
            m.calibrate(constant=constant, method=method, robust=robust, log_lik_tol=log_lik_tol)
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
        df = int(self.thresholds_uncentred.notna().sum().sum()) - 1

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
        omnibus_scope="item",
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
        method="log-lik",
        robust=False,
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
           estimated by this method — see SLM's dif_test docstring for
           the exact per-item LR mechanics (H1 = each group's own
           natively-calibrated fit, computed once; H0_i = item i's
           location pooled across groups, precision-weighted, with that
           item's row in thresholds_uncentred rebuilt from the pooled
           value plus that group's own centred category shape for the
           item, then that group's person locations re-estimated). Results in
           self.dif_table and self.dif_omnibus_table.

        2. Threshold-structure DIF (DCF, differential category functioning):
           tested via each item's category
           *widths* (self.thresholds[k+1] - self.thresholds[k]), not raw
           threshold locations — locations are partial sums of widths, so
           a single genuine width change cascades into every downstream
           threshold location, smearing/mis-localising the signal if
           tested directly; widths isolate it to the one category that
           actually changed. Like RSM's shared threshold vector, these are
           not translated by the item-scale purification in (1). Each
           item's widths are Wald-tested category-by-category, reference
           vs. focal, using cat_width_se (from std_errors() — differenced
           *within* each bootstrap resample before taking the std, so it
           already reflects Cov(threshold_k, threshold_{k+1}) without a
           separate covariance term). Each item's own category-width tests
           form their own multiple-comparison correction pool, independent
           of other items and of the item-location correction pool in (1).
           Not affected by test=/omnibus= — those only govern component
           (1). Results in self.threshold_dif_table.

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
            re-optimise nuisance person location parameters independently, which
            inflates the LR statistic beyond what df=k-1 accounts for. This
            alone is enough for SLM/RSM (confirmed by null-DIF simulation:
            ~1.3-1.7x nominal Type I error with own-group person locations, ~nominal
            with pooled person locations) but NOT for PCM — see omnibus_scope.
        omnibus_scope : {'item', 'full'}, default 'item'
            Unlike SLM (no thresholds) or RSM (one shared threshold vector
            for the whole test), PCM has a separate threshold/category-
            structure per item. Each group's own thresholds_uncentred still differs
            freely from the pooled model's even after pooling person locations,
            adding items*(max_score-1) unaccounted-for free parameters to
            the H1 side — confirmed by null-DIF simulation to inflate the
            omnibus test massively (~0.6 rejection rate at nominal 0.05,
            essentially unrelated to the person-location-pooling fix above, which
            barely moves it). Two ways to correctly account for this,
            corresponding to the classic uniform-vs-non-uniform DIF
            distinction:
            'item' (default) — an omnibus test of uniform DIF only.
            Isolates item-location DIF specifically, matching SLM/RSM's
            df=k-1 scope. Holds both person locations AND each item's category
            step-shape fixed at the pooled model's values, then re-optimises
            only a single location shift per item via Newton-Raphson
            (mirroring person()'s location-estimation loop, see
            _item_location_log_likelihood) — this isolates exactly the
            item-location parameter difference the df already accounts for.
            A constant group difference in log-odds at every category
            transition is the textbook definition of uniform DIF. Confirmed
            by null-DIF simulation: ~nominal rejection rate. Slower than
            'full' (one Newton-Raphson pass per focal group, similar cost to
            an extra person-location re-estimation).
            'full' — a joint omnibus test of uniform DIF *and* differential
            category functioning (DCF), the Rasch/PCM-family analogue of
            non-uniform DIF (Rasch-family models fix discrimination equal
            across groups by construction, so there's no literal
            group x person-location interaction slope to test the way 2PL/logistic-
            regression DIF methods do — DCF, i.e. the category/threshold
            structure itself differing by group so the group gap varies
            across categories, is how a non-uniform-like effect manifests
            here instead). Cheaper alternative — keeps each group's own
            independently-fit thresholds_uncentred (no extra re-estimation
            needed) but widens df to (k-1) + sum(max_score_vector - 1) so
            the full threshold freedom is properly charged for. Important:
            this is a single joint p-value, not a decomposition — a
            significant 'full' result alongside a non-significant 'item'
            result suggests the DIF is predominantly category-structure-
            driven, but 'full' alone can't isolate or quantify that split.
            For that, use dif_table (item-location / uniform) and
            threshold_dif_table (per-item category widths / the DCF
            component) directly, each independently Wald-tested and
            corrected — omnibus_scope only
            controls the scope of this one joint significance test, not
            what dif_table/threshold_dif_table report. Confirmed by
            null-DIF simulation: ~nominal rejection rate, slightly
            conservative.
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
            threshold-cell — same scope as welch, unlike category below)
            to what it would be at a standard reference sample size
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
            each focal-group comparison) and, separately, across every
            item x category cell (within each focal-group comparison).
        alpha : float, default 0.05
            p-value threshold for flagging (Wald, per-item LR, and
            omnibus alike). Compared against the corrected p-value where
            correction applies.
        logit_threshold : float, default 0.43
            Absolute logit-difference threshold for flagging (ETS-style
            convention), applied to the Wald test in both components. An
            item/threshold cell is flagged (Flagged or Flagged_LR) only if
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
            reference in one call, and auto-rendering several plot windows
            at once isn't
            the right default.
        plot_kwargs : dict or None, default None
            Extra keyword arguments forwarded to plot_anchor_selection()
            for every focal group when plot=True (e.g. filename, xmin/
            xmax, title).
        warm_corr, tolerance, max_iters, ext_score_adjustment : floats
            Person estimation kwargs, passed through to group models. Only
            actually used when test in ('lr', 'both') or omnibus=True.
        constant, method, matrix_power, log_lik_tol : floats
            Calibration kwargs passed to group models. matrix_power is
            accepted for signature consistency with SLM/RSM but unused —
            PCM's calibrate() does not take a matrix_power argument.

        Attributes set
        --------------
        dif_table : pandas.DataFrame
            One row per (item, focal group) pair. Columns: 'Group'
            (focal group value), 'Reference' / 'Focal' / 'Focal (purified)'
            (central item location estimates), 'Difference' (purified focal -
            reference), 'SE', 'z', 'p', 'p (corrected)', 'Selected' (used
            to define the purified scale), 'Flagged' (Wald p and logit-
            difference thresholds both met). If welch=True, also 'df'
            (Welch-Satterthwaite), and 'p'/'p (corrected)' are t-based
            rather than normal-based. If test is 'lr' or 'both', also:
            'LR', 'p_LR', 'p_LR (corrected)', 'Flagged_LR'. If
            category=True, also 'Category' ('A', 'B+', 'B-', 'C+', or
            'C-').
        dif_omnibus_table : pandas.DataFrame or None
            One row per focal group: 'LR', 'df', 'p', 'Flagged' — Andersen-
            style joint test of every common item at once. Item-location
            component only if omnibus_scope='item' (default); item location
            and threshold/category-structure jointly if omnibus_scope='full'
            — see omnibus_scope. None if omnibus=False.
        threshold_dif_table : pandas.DataFrame
            Differential category functioning (DCF), tested via category
            *widths* (thresholds[k+1] - thresholds[k]) rather than raw
            threshold locations — locations are partial sums of widths, so
            a single genuine width change cascades into every downstream
            threshold location; widths localise the signal to the one
            category that actually changed instead. One row per (item,
            category, focal group), indexed by ('Item', 'Category') where
            Category k is the width between threshold k and k+1 (so an
            item with max_score thresholds has max_score-1 rows). Columns:
            'Group', 'Reference' / 'Focal' (category widths), 'Difference',
            'SE' (from cat_width_se — bootstrap SE of the width itself,
            i.e. differenced *within* each resample before taking the std,
            so it already reflects Cov(threshold_k, threshold_{k+1})), 'z',
            'p', 'p (corrected)', 'Flagged'. Corrected independently of
            dif_table, per item. If welch=True, also 'df'
            (Welch-Satterthwaite), and 'p'/'p (corrected)' are t-based
            rather than normal-based. Items whose max_score differs
            between the reference and a focal group, or whose max_score is
            1 (no category width to test), are excluded from this table
            for that group — the former with a UserWarning.
        dif_reference : the reference group value.
        dif_covariate : the covariate column name used.
        dif_reference_model : PCM
            Fitted model for the reference group.
        dif_focal_models : dict {focal_value: PCM}
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
        from raschpy.pcm import PCM

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
        max_scores = self.responses.notna().mul(self.max_score_vector, axis=1).sum(axis=1)
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
        ref_model = PCM(self.responses.loc[ref_idx], self.max_score_vector)
        ref_model.calibrate(constant=constant, method=method, robust=robust, log_lik_tol=log_lik_tol)
        ref_model.std_errors(
            no_of_samples=no_of_samples, constant=constant, method=method,
            log_lik_tol=log_lik_tol, seed=seed,
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
            focal_model = PCM(self.responses.loc[focal_idx], self.max_score_vector)
            focal_model.calibrate(constant=constant, method=method, robust=robust, log_lik_tol=log_lik_tol)
            focal_model.std_errors(
                no_of_samples=no_of_samples, constant=constant, method=method,
                log_lik_tol=log_lik_tol, seed=seed,
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
                m_full = PCM(self.responses.loc[combined_idx], self.max_score_vector)
                m_full.calibrate(constant=constant, method=method, robust=robust, log_lik_tol=log_lik_tol)
                m_full.person_estimates(**pe_kw)
                ll_full = m_full._log_likelihood()
                pooled_ref_persons = m_full.persons.reindex(ref_model.responses.index)
                pooled_focal_persons = m_full.persons.reindex(focal_model.responses.index)
                if omnibus_scope == "item":
                    pooled_step_shape = m_full.thresholds_uncentred.sub(m_full.items, axis=0)
                    ll_ref_omni = self._item_location_log_likelihood(
                        ref_model.responses, pooled_ref_persons,
                        pooled_step_shape, list(common_items),
                        tolerance=tolerance, max_iters=max_iters,
                    )
                    ll_focal_omni = self._item_location_log_likelihood(
                        focal_model.responses, pooled_focal_persons,
                        pooled_step_shape, list(common_items),
                        tolerance=tolerance, max_iters=max_iters,
                    )
                    df_omni = len(common_items) - 1
                else:
                    ll_ref_omni = ref_model._log_likelihood(persons=pooled_ref_persons)
                    ll_focal_omni = focal_model._log_likelihood(persons=pooled_focal_persons)
                    df_omni = (
                        len(common_items) - 1
                        + sum(int(self.max_score_vector[item]) - 1 for item in common_items)
                    )
                lr_omni = max(0.0, -2 * (ll_full - (ll_ref_omni + ll_focal_omni)))
                p_omni = float(chi2.sf(lr_omni, df_omni))
                omnibus_rows[focal] = {
                    "LR": lr_omni, "df": df_omni, "p": p_omni,
                    "Flagged": p_omni < alpha,
                }

            if test in ("lr", "both"):
                ll_h1 = ll_ref + ll_focal
                ref_scratch = PCM(ref_model.responses, self.max_score_vector)
                focal_scratch = PCM(focal_model.responses, self.max_score_vector)
                lr_rows = {}
                for item in common_items:
                    w_ref = 1.0 / ref_model.item_se[item] ** 2
                    w_focal = 1.0 / focal_model.item_se[item] ** 2
                    pooled = (
                        w_ref * ref_model.items[item] + w_focal * focal_shifted[item]
                    ) / (w_ref + w_focal)
                    pooled_focal_scale = pooled - tc

                    ref_scratch.thresholds_uncentred = ref_model.thresholds_uncentred.copy()
                    ref_offsets = ref_model.thresholds.loc[item].dropna()
                    row = np.full(ref_scratch.thresholds_uncentred.shape[1], np.nan)
                    row[: len(ref_offsets)] = pooled + ref_offsets.values
                    ref_scratch.thresholds_uncentred.loc[item] = row
                    ref_scratch.person_estimates(**pe_kw)
                    ll_ref_h0 = ref_scratch._log_likelihood()

                    focal_scratch.thresholds_uncentred = (
                        focal_model.thresholds_uncentred.copy()
                    )
                    focal_offsets = focal_model.thresholds.loc[item].dropna()
                    row_f = np.full(focal_scratch.thresholds_uncentred.shape[1], np.nan)
                    row_f[: len(focal_offsets)] = pooled_focal_scale + focal_offsets.values
                    focal_scratch.thresholds_uncentred.loc[item] = row_f
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

            # --- Threshold-structure DIF (DCF: category widths, per item, no
            # purification) ---
            # Category k's width (thresholds[k+1] - thresholds[k]) is the natural unit
            # for category-structure DIF, not raw threshold *locations* — locations are
            # partial sums of widths, so a single genuine width change cascades into
            # every downstream threshold location, smearing/mis-localising the signal
            # if tested directly. cat_width_se (from std_errors()) differences *within*
            # each bootstrap resample before taking the std, so it already correctly
            # reflects Cov(threshold_k, threshold_{k+1}) with no extra covariance term
            # needed.
            thr_rows = []
            for item in common_items:
                ref_thr = ref_model.thresholds.loc[item].dropna()
                focal_thr = focal_model.thresholds.loc[item].dropna()
                if len(ref_thr) != len(focal_thr):
                    warnings.warn(
                        f"Item '{item}' has a different number of categories "
                        f"between the reference and '{focal}' groups "
                        f"({len(ref_thr)} vs {len(focal_thr)}); "
                        f"threshold DIF skipped for this item in this "
                        f"comparison.",
                        UserWarning,
                        stacklevel=2,
                    )
                    continue
                if len(ref_thr) < 2:
                    continue  # single-threshold (binary) item — no category width to test
                ref_widths = ref_thr.diff().dropna()
                focal_widths = focal_thr.diff().dropna()
                ref_widths.index = focal_widths.index = range(1, len(ref_widths) + 1)
                ref_se = ref_model.cat_width_se[item]
                focal_se = focal_model.cat_width_se[item]
                for k in ref_widths.index:
                    ref_val = ref_widths.loc[k]
                    focal_val = focal_widths.loc[k]
                    se_val = np.sqrt(ref_se.loc[k] ** 2 + focal_se.loc[k] ** 2)
                    thr_rows.append(
                        {
                            "Item": item,
                            "Category": k,
                            "Group": focal,
                            "Reference": ref_val,
                            "Focal": focal_val,
                            "Difference": focal_val - ref_val,
                            "SE": se_val,
                            "_ref_se": ref_se.loc[k],
                            "_focal_se": focal_se.loc[k],
                        }
                    )

            threshold_table = pd.DataFrame(thr_rows)
            if not threshold_table.empty:
                ref_n_thr = ref_model.no_of_persons
                focal_n_thr = focal_model.no_of_persons
                if size_adjust:
                    threshold_table["_ref_se"] = threshold_table["_ref_se"] * np.sqrt(
                        ref_n_thr / reference_n
                    )
                    threshold_table["_focal_se"] = threshold_table["_focal_se"] * np.sqrt(
                        focal_n_thr / reference_n
                    )
                    threshold_table["SE"] = np.sqrt(
                        threshold_table["_ref_se"] ** 2 + threshold_table["_focal_se"] ** 2
                    )
                    ref_n_thr = focal_n_thr = reference_n

                if welch:
                    thr_z_vals, thr_df_vals, thr_p_vals = self._welch_satterthwaite(
                        threshold_table["Difference"].values,
                        threshold_table["_ref_se"].values, ref_n_thr,
                        threshold_table["_focal_se"].values, focal_n_thr,
                    )
                    threshold_table["z"] = thr_z_vals
                    threshold_table["df"] = thr_df_vals
                    threshold_table["p"] = thr_p_vals
                else:
                    threshold_table["z"] = (
                        threshold_table["Difference"] / threshold_table["SE"]
                    )
                    threshold_table["p"] = 2 * norm.sf(np.abs(threshold_table["z"]))
                threshold_table = threshold_table.drop(columns=["_ref_se", "_focal_se"])

                # Correction pool is per-item (each item's own max_score threshold
                # tests form their own family), not global across every item x
                # threshold pair — mirrors dif_table's "all items" family for
                # item-location DIF, and keeps power to detect genuine per-item DCF
                # from collapsing as the number of items grows.
                if correction == "bonferroni":
                    group_sizes = threshold_table.groupby("Item")["p"].transform("size")
                    threshold_table["p (corrected)"] = (
                        threshold_table["p"] * group_sizes
                    ).clip(upper=1.0)
                elif correction == "bh":
                    threshold_table["p (corrected)"] = threshold_table.groupby("Item")[
                        "p"
                    ].transform(lambda s: self._bh_correction(s).values)
                else:
                    threshold_table["p (corrected)"] = threshold_table["p"]

                threshold_table["Flagged"] = (
                    threshold_table["p (corrected)"] < alpha
                ) & (threshold_table["Difference"].abs() >= logit_threshold)

            threshold_table = threshold_table.set_index(["Item", "Category"])

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
        self.threshold_dif_table = pd.concat(all_threshold_rows)
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
        method="log-lik",
        robust=False,
        log_lik_tol=0.000001,
        seed=None,
    ):
        """
        Compare PCM against RSM using a likelihood ratio test, AIC, or BIC.

        RSM is the constrained (nested) model; PCM is unconstrained. Requires
        uniform max scores across items.

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
        from raschpy.rsm import RSM

        if test not in ("LR", "AIC", "BIC"):
            raise ValueError("test must be 'LR', 'AIC', or 'BIC'")
        if sampling is not None and sampling != "dynamic" and not isinstance(sampling, int):
            raise ValueError("sampling must be None, 'dynamic', or an integer")

        if not (self.max_score_vector == self.max_score_vector.iloc[0]).all():
            raise ValueError(
                "RSM vs PCM comparison requires uniform max scores across items."
            )
        max_score = int(self.max_score_vector.iloc[0])

        if not hasattr(self, "thresholds_uncentred"):
            self.calibrate(constant=constant, method=method, robust=robust, log_lik_tol=log_lik_tol)
        if not hasattr(self, "persons"):
            self.person_estimates(
                warm_corr=warm_corr,
                tolerance=tolerance,
                max_iters=max_iters,
                ext_score_adjustment=ext_score_adjustment,
            )

        scores = self.responses.sum(axis=1)
        max_scores = self.responses.notna().sum(axis=1) * max_score
        non_extreme_mask = (scores > 0) & (scores < max_scores)
        n_persons = int(non_extreme_mask.sum())

        # Fit RSM on full data (parameters used for both full and sampled LL)
        rsm = RSM(self.responses)
        rsm.calibrate(constant=constant, method=method, log_lik_tol=log_lik_tol)
        rsm.person_estimates(
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
                min(20 * (self.no_of_items - 1) * (max_score - 1), 1500)
                if sampling == "dynamic"
                else int(sampling)
            )
            if n_persons > T:
                rng = np.random.default_rng(seed)
                non_extreme_idx = self.responses.index[non_extreme_mask]
                sampled_idx = rng.choice(non_extreme_idx, size=T, replace=False)
                ll_responses = self.responses.loc[sampled_idx]
                n_ll = T

        ll_pcm = self._log_likelihood(responses=ll_responses)
        ll_rsm = rsm._log_likelihood(responses=ll_responses)

        k_pcm = int(self.thresholds_uncentred.notna().sum().sum()) - 1
        k_rsm = (self.no_of_items - 1) + (max_score - 1)

        effect = float(self.thresholds.sub(rsm.thresholds, axis=1).abs().to_numpy().max())
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
            df = (self.no_of_items - 1) * (max_score - 1)
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
        method="log-lik",
        robust=False,
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
            Newton-Raphson convergence tolerance.
        max_iters : int, default 100
            Maximum iterations.
        ext_score_adjustment : float, default 0.5
            Extreme score adjustment.
        constant : float, default 0.1
            Additive smoothing constant.
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
                robust=robust,
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
            self.eigenvectors = None
            self.eigenvalues = None
            self.variance_explained = None
            self.loadings = None

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
        robust=False,
        matrix_power=None,
        log_lik_tol=0.000001,
        no_of_samples=500,
        interval=None,
        seed=None,
    ):
        """
        Build and store the item statistics summary table.

        Auto-triggers std_errors() and fit_statistics() if not yet run.
        Always includes central item location estimates, response counts,
        facilities, and Infit/Outfit MS. Additional columns (SE, Z
        statistics, point-measure correlations, CI bounds) are included
        based on flags or when full=True.

        Parameters
        ----------
        full : bool, default False
            If True, sets zstd=True, point_measure_corr=True, and
            interval=0.95. Overrides individual flags.
        zstd : bool, default False
            If True, includes Infit Z and Outfit Z columns.
        point_measure_corr : bool, default False
            If True, includes observed and expected point-measure
            correlation columns.
        dp : int, default 3
            Number of decimal places for rounding numeric output.
        se : bool, default True
            If True, computes and includes the SE column (and CI bound
            columns, if interval is set). If False, skips the bootstrap
            entirely — useful when only Infit/Outfit MS are needed (e.g.
            repeated simulation runs), since those do not depend on the
            bootstrap. Forces interval to None when False.
        warm_corr : bool, default True
            Warm bias correction for person location estimates.
        tolerance : float, default 0.00001
            Newton-Raphson convergence tolerance.
        max_iters : int, default 100
            Maximum Newton-Raphson iterations.
        ext_score_adjustment : float, default 0.5
            Extreme score adjustment for person location estimation.
        method : str, default 'log-lik'
            Priority vector extraction method for calibration.
        constant : float, default 0.1
            Additive smoothing constant for calibration.
        matrix_power : int or None, default None
            Matrix power for calibration. None resolves to 0 for
            method='log-lik' (no powering), else 5.
        log_lik_tol : float, default 0.000001
            Log-likelihood tolerance for calibration.
        no_of_samples : int, default 500
            Bootstrap samples for SE estimation. Unused if se=False.
        interval : float or None, default None
            Confidence interval width (e.g. 0.95). If provided, lower
            and upper percentile columns are included. Ignored if se=False.
        seed : int or None, default None
            Seed passed through to the internal std_errors()/fit_statistics()
            calls (only used if not already computed). None draws fresh
            entropy each call.

        Attributes set
        --------------
        item_stats : pandas.DataFrame
            Item statistics table with items as rows. Always contains
            'Estimate', 'Count', 'Facility', 'Infit MS', 'Outfit MS'.
            Optional: 'SE' and CI bounds (if se=True), 'Infit Z',
            'Outfit Z', 'PM corr', 'Exp PM corr'.

        Returns
        -------
        None
        """

        if full:
            zstd = True
            point_measure_corr = True
            if interval is None:
                interval = 0.95

        if not se:
            interval = None

        if se and (
            not hasattr(self, "threshold_low")
            or self.threshold_low is None
            and interval is not None
        ):
            self.std_errors(
                interval=interval,
                no_of_samples=no_of_samples,
                constant=constant,
                robust=robust,
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
                matrix_power=matrix_power,
                log_lik_tol=log_lik_tol,
                no_of_samples=no_of_samples,
                interval=interval,
                seed=seed,
            )

        stats = pd.DataFrame(index=self.item_names)
        stats["Estimate"] = self.items.round(dp)
        if se:
            stats["SE"] = self.item_se.round(dp)
            if interval is not None:
                stats[f"{round((1 - interval) * 50, 1)}%"] = self.central_low.round(dp)
                stats[f"{round((1 + interval) * 50, 1)}%"] = self.central_high.round(dp)
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
        zstd=True,
        disc=False,
        point_measure_corr=False,
        dp=3,
        warm_corr=True,
        tolerance=0.00001,
        max_iters=100,
        ext_score_adjustment=0.5,
        method="log-lik",
        constant=None,
        robust=False,
        no_of_samples=500,
        interval=None,
    ):
        """
        Build and store the threshold statistics summary tables.

        Auto-triggers fit_statistics() if not yet run. Produces two tables:
        uncentred thresholds (absolute logit values) and centred thresholds
        (offsets from the central item location). Both always include
        estimates, SEs, and Infit/Outfit MS. Additional columns are included
        based on flags or when full=True.

        Parameters
        ----------
        full : bool, default False
            If True, sets zstd=True, disc=True, point_measure_corr=True,
            and interval=0.95.
        zstd : bool, default True
            If True, includes Infit Z and Outfit Z columns.
        disc : bool, default False
            If True, includes threshold discrimination column.
        point_measure_corr : bool, default False
            If True, includes observed and expected point-measure
            correlation columns.
        dp : int, default 3
            Number of decimal places for rounding numeric output.
        warm_corr : bool, default True
            Warm bias correction for person location estimates.
        tolerance : float, default 0.00001
            Newton-Raphson convergence tolerance.
        max_iters : int, default 100
            Maximum Newton-Raphson iterations.
        ext_score_adjustment : float, default 0.5
            Extreme score adjustment for person location estimation.
        method : str, default 'log-lik'
            Priority vector extraction method for calibration.
        constant : float, default 0.1
            Additive smoothing constant for calibration.
        no_of_samples : int, default 500
            Bootstrap samples for SE estimation.
        interval : float or None, default None
            Confidence interval width. If provided, lower and upper
            percentile columns are included.

        Attributes set
        --------------
        threshold_stats_uncentred : pandas.DataFrame
            Threshold statistics using absolute (uncentred) threshold values.
            MultiIndex rows (item, threshold number).
        threshold_stats : pandas.DataFrame
            Threshold statistics using centred threshold offsets (uncentred
            estimate minus central item location). See also
            category_stats_df() for category-*width* statistics — the
            physically meaningful, full-rank quantity for step-structure
            questions (each item's zero-summed threshold vector only has
            max_score_vector[item]-1 true degrees of freedom, so per-
            threshold SEs here are correlated, not independent).

        Returns
        -------
        None
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
                no_of_samples=no_of_samples,
                interval=interval,
            )

        estimate_array = np.concatenate(
            [
                self.thresholds_uncentred.loc[item].dropna().values
                for item in self.responses.columns
            ]
        )
        se_array = np.concatenate(
            [self.threshold_se[item] for item in self.responses.columns]
        )

        stats = pd.DataFrame(index=self.threshold_infit_ms.index)
        stats["Estimate"] = estimate_array.round(dp)
        stats["SE"] = se_array.round(dp)

        if interval is not None:
            low_array = np.concatenate(
                [self.threshold_low[item] for item in self.responses.columns]
            )
            high_array = np.concatenate(
                [self.threshold_high[item] for item in self.responses.columns]
            )
            stats[f"{round((1 - interval) * 50, 1)}%"] = low_array.round(dp)
            stats[f"{round((1 + interval) * 50, 1)}%"] = high_array.round(dp)

        stats["Infit MS"] = (
            self.threshold_infit_ms.reset_index(drop=True).round(dp).values
        )
        if zstd:
            stats["Infit Z"] = (
                self.threshold_infit_zstd.reset_index(drop=True).round(dp).values
            )
        stats["Outfit MS"] = (
            self.threshold_outfit_ms.reset_index(drop=True).round(dp).values
        )
        if zstd:
            stats["Outfit Z"] = (
                self.threshold_outfit_zstd.reset_index(drop=True).round(dp).values
            )
        if disc:
            stats["Discrim"] = (
                self.threshold_discrimination.reset_index(drop=True).round(dp).values
            )
        if point_measure_corr:
            stats["PM corr"] = (
                self.threshold_point_measure.reset_index(drop=True).round(dp).values
            )
            stats["Exp PM corr"] = (
                self.threshold_exp_point_measure.reset_index(drop=True).round(dp).values
            )

        self.threshold_stats_uncentred = stats

        central_array = np.concatenate(
            [
                np.full(self.max_score_vector[item], self.items[item])
                for item in self.responses.columns
            ]
        )
        self.threshold_stats = stats.copy()
        self.threshold_stats["Estimate"] -= central_array.round(dp)
        if interval is not None:
            self.threshold_stats[
                f"{round((1 - interval) * 50, 1)}%"
            ] -= central_array.round(dp)
            self.threshold_stats[
                f"{round((1 + interval) * 50, 1)}%"
            ] -= central_array.round(dp)

    def category_stats_df(
        self,
        dp=3,
        constant=None,
        method="log-lik",
        robust=False,
        matrix_power=None,
        log_lik_tol=0.000001,
        no_of_samples=500,
        interval=None,
        seed=None,
    ):
        """
        Build and store the category width statistics summary table.

        Reports each item's category *widths* (thresholds[k+1] -
        thresholds[k]) rather than raw threshold locations — see
        threshold_stats_df for the full rationale: each item's zero-summed
        threshold vector has only max_score_vector[item]-1 true degrees of
        freedom, so the per-threshold SEs threshold_stats_df reports are
        correlated, not independent, and understate the true uncertainty
        of the physically meaningful step-structure quantity. Reported
        *alongside* threshold_stats_df's output, not instead of it —
        threshold-level SEs remain the expected, standard report. Widths
        are identical whether computed from centred or uncentred
        thresholds (a constant per-item shift cancels in the difference),
        so there is only one category_stats table, not a centred/uncentred
        pair.

        Deliberately lighter than threshold_stats_df: no Infit/Outfit or
        other fit statistics, since those aren't naturally defined for a
        difference of two threshold locations. Auto-triggers calibrate()/
        std_errors() directly if not yet run (not the full, heavier
        fit_statistics()).

        Widths can be negative — a negative width at category k means
        thresholds k and k+1 are disordered for that item (category k is
        never the most likely response at any person location level). Prop
        disordered makes this a continuous diagnostic rather than a
        single point-estimate yes/no: the proportion of bootstrap
        resamples in which that item's category width was negative —
        reasonably read as the probability that the true category is
        disordered.

        Parameters
        ----------
        dp : int, default 3
            Decimal places.
        constant, method, matrix_power, log_lik_tol : floats
            Calibration kwargs, used only if calibrate() hasn't already
            been run (matrix_power is accepted for signature consistency
            but PCM's own calibrate() does not take it).
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
            MultiIndex rows (item, category number), one fewer row per
            item than threshold_stats. Items with max_score_vector==1
            (binary — no category width to report) are excluded. Columns:
            Estimate (the width itself, can be negative), SE (from
            cat_width_se — differenced *within* each bootstrap resample
            before taking the std, so it already reflects
            Cov(threshold_k, threshold_{k+1})), CI bounds if interval is
            not None, Disordered (Estimate < 0), and Prop disordered — the
            bootstrap proportion of resamples with a negative width,
            reasonably read as the probability that the true category is
            disordered. Useful for interpreting Disordered in both
            directions: a True that's only weakly supported (Prop
            disordered close to 0.5) vs. robust, or a False that's
            nonetheless uncertain (Prop disordered not small) vs.
            clear-cut.
        """
        if not hasattr(self, "thresholds"):
            self.calibrate(constant=constant, method=method, robust=robust, log_lik_tol=log_lik_tol)
        if not hasattr(self, "cat_width_se"):
            self.std_errors(
                interval=interval, no_of_samples=no_of_samples,
                constant=constant, method=method, matrix_power=matrix_power,
                log_lik_tol=log_lik_tol, seed=seed,
            )

        cat_items = [
            item for item in self.responses.columns
            if len(self.thresholds.loc[item].dropna()) >= 2
        ]
        if cat_items:
            cat_widths = {
                item: self.thresholds.loc[item].dropna().diff().dropna().values
                for item in cat_items
            }
            cat_mi = pd.MultiIndex.from_tuples(
                [(item, k) for item in cat_items for k in range(1, len(cat_widths[item]) + 1)]
            )
            stats = pd.DataFrame(index=cat_mi)
            stats["Estimate"] = np.concatenate(
                [cat_widths[item] for item in cat_items]
            ).round(dp)
            stats["SE"] = np.concatenate(
                [self.cat_width_se[item].values for item in cat_items]
            ).round(dp)
            if interval is not None and self.cat_width_low is not None:
                stats[f"{round((1 - interval) * 50, 1)}%"] = np.concatenate(
                    [self.cat_width_low[item] for item in cat_items]
                ).round(dp)
                stats[f"{round((1 + interval) * 50, 1)}%"] = np.concatenate(
                    [self.cat_width_high[item] for item in cat_items]
                ).round(dp)
            stats["Disordered"] = np.concatenate(
                [cat_widths[item] for item in cat_items]
            ) < 0
            stats["Prop disordered"] = np.concatenate(
                [
                    (self.cat_width_bootstrap[item] < 0).mean(axis=0).values
                    for item in cat_items
                ]
            ).round(dp)
        else:
            stats = pd.DataFrame(columns=["Estimate", "SE", "Disordered", "Prop disordered"])
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
        robust=False,
    ):
        """
        Build and store the person statistics summary table.

        Auto-triggers fit_statistics() if not yet run. One row per person
        with person location estimate, CSEM, raw score, max score, proportion correct,
        and Infit/Outfit MS and Z statistics.

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
            Newton-Raphson convergence tolerance.
        max_iters : int, default 100
            Maximum Newton-Raphson iterations.
        ext_score_adjustment : float, default 0.5
            Extreme score adjustment.
        method : str, default 'log-lik'
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
        if not hasattr(self, "person_infit_ms"):
            self.fit_statistics(
                se=se,
                warm_corr=warm_corr,
                tolerance=tolerance,
                max_iters=max_iters,
                ext_score_adjustment=ext_score_adjustment,
                method=method,
                constant=constant,
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
        stats["Max score"] = (
            self.responses.notna()
            .astype(int)
            .mul(self.max_score_vector, axis=1)
            .sum(axis=1)
            .astype(int)
        )
        stats["p"] = (stats["Score"] / stats["Max score"]).round(dp)

        # BUG FIX: original used .update(dict) which ignores index alignment.
        # Direct assignment aligns on index correctly.
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
        robust=False,
        alpha=False,
        seed=None,
    ):
        """
        Build and store the test-level summary statistics table.

        Auto-triggers fit_statistics() if not yet run. Produces a compact
        three-column table (Items, Thresholds, Persons) covering mean, SD,
        separation ratio, strata, and reliability.

        Parameters
        ----------
        dp : int, default 3
            Number of decimal places for rounding numeric output.
        warm_corr : bool, default True
            Warm bias correction for person location estimates.
        tolerance : float, default 0.00001
            Newton-Raphson convergence tolerance.
        max_iters : int, default 100
            Maximum Newton-Raphson iterations.
        ext_score_adjustment : float, default 0.5
            Extreme score adjustment for person location estimation.
        method : str, default 'log-lik'
            Priority vector extraction method for calibration.
        constant : float, default 0.1
            Additive smoothing constant for calibration.
        alpha : bool, default False
            If True, adds a 'Cronbach alpha' row (Persons column only —
            Items and Thresholds are left NaN, since Cronbach's alpha is a
            person-side reliability statistic). Computed on complete cases;
            if the data contain missing responses, a UserWarning is raised
            and alpha is computed after listwise deletion, which may
            underestimate the true value.
        seed : int or None, default None
            Seed passed through to the internal fit_statistics() call (only
            used if not already computed). None draws fresh entropy.

        Attributes set
        --------------
        test_stats : pandas.DataFrame
            Three-column table (Items, Thresholds, Persons) with rows:
            Mean, SD, Separation ratio, Strata, Reliability.

        Returns
        -------
        None
        """

        if not hasattr(self, "psi"):
            self.fit_statistics(
                warm_corr=warm_corr,
                tolerance=tolerance,
                max_iters=max_iters,
                ext_score_adjustment=ext_score_adjustment,
                method=method,
                constant=constant,
                seed=seed,
            )

        items_col = [self.items.mean(), self.items.std(), self.isi_central,
                     self.item_strata, self.item_reliability]
        thr_col = [self.threshold_list.mean(), self.threshold_list.std(),
                   self.isi_thresholds, self.threshold_strata, self.threshold_reliability]
        persons_col = [self.persons.mean(), self.persons.std(), self.psi,
                       self.person_strata, self.person_reliability]
        index = ["Mean", "SD", "Separation ratio", "Strata", "Reliability"]

        if alpha:
            items_col.append(np.nan)
            thr_col.append(np.nan)
            persons_col.append(self._cronbach_alpha())
            index.append("Cronbach alpha")

        self.test_stats = pd.DataFrame(
            {"Items": items_col, "Thresholds": thr_col, "Persons": persons_col},
            index=index,
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
        robust=False,
        no_of_samples=500,
        interval=None,
    ):
        """
        Export item, threshold, person, and test statistics to file.

        Auto-triggers item_stats_df(), threshold_stats_df(), person_stats_df(),
        and test_stats_df() if not yet run. Saves all tables to either a single
        Excel workbook (one sheet per table) or separate CSV files.

        Parameters
        ----------
        filename : str
            Output filename or path (without extension for CSV; '.xlsx'
            appended automatically for Excel format).
        format : str, default 'csv'
            Output format: 'csv' saves five separate CSV files; 'xlsx'
            saves all tables to sheets in a single Excel workbook.
        dp : int, default 3
            Decimal places for rounding.
        warm_corr : bool, default True
            Warm bias correction. Passed to stats_df methods if needed.
        tolerance : float, default 0.00001
            Newton-Raphson convergence tolerance.
        max_iters : int, default 100
            Maximum Newton-Raphson iterations.
        ext_score_adjustment : float, default 0.5
            Extreme score adjustment.
        method : str, default 'log-lik'
            Priority vector extraction method.
        constant : float, default 0.1
            Additive smoothing constant.
        no_of_samples : int, default 500
            Bootstrap samples for SE estimation.
        interval : float or None, default None
            Confidence interval width.

        Returns
        -------
        None
        """

        if not hasattr(self, "item_stats"):
            self.item_stats_df(
                dp=dp,
                warm_corr=warm_corr,
                tolerance=tolerance,
                max_iters=max_iters,
                ext_score_adjustment=ext_score_adjustment,
                method=method,
                constant=constant,
                no_of_samples=no_of_samples,
                interval=interval,
            )
        if not hasattr(self, "threshold_stats_uncentred"):
            self.threshold_stats_df(
                dp=dp,
                warm_corr=warm_corr,
                tolerance=tolerance,
                max_iters=max_iters,
                ext_score_adjustment=ext_score_adjustment,
                method=method,
                constant=constant,
                no_of_samples=no_of_samples,
                interval=interval,
            )
        if not hasattr(self, "person_stats"):
            self.person_stats_df(
                dp=dp,
                warm_corr=warm_corr,
                tolerance=tolerance,
                max_iters=max_iters,
                ext_score_adjustment=ext_score_adjustment,
                method=method,
                constant=constant,
            )
        if not hasattr(self, "test_stats"):
            self.test_stats_df(
                dp=dp,
                warm_corr=warm_corr,
                tolerance=tolerance,
                max_iters=max_iters,
                ext_score_adjustment=ext_score_adjustment,
                method=method,
                constant=constant,
            )

        if format == "xlsx":
            if not filename.endswith(".xlsx"):
                filename += ".xlsx"
            with pd.ExcelWriter(
                filename, engine="openpyxl"
            ) as writer:  # BUG FIX: writer.save() deprecated
                self.item_stats.to_excel(writer, sheet_name="Item statistics")
                self.threshold_stats_uncentred.to_excel(
                    writer, sheet_name="Threshold statistics uncentred"
                )
                self.threshold_stats.to_excel(
                    writer, sheet_name="Threshold statistics centred"
                )
                self.person_stats.to_excel(writer, sheet_name="Person statistics")
                self.test_stats.to_excel(writer, sheet_name="Test statistics")
        else:
            if filename.endswith(".csv"):
                filename = filename[:-4]
            self.item_stats.to_csv(f"{filename}_item_stats.csv")
            self.threshold_stats_uncentred.to_csv(
                f"{filename}_threshold_stats_uncentred.csv"
            )
            self.threshold_stats.to_csv(f"{filename}_threshold_stats_centred.csv")
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
        robust=False,
    ):
        """
        Export residual correlation analysis results to file.

        Auto-triggers res_corr_analysis() if not yet run. Saves eigenvectors,
        eigenvalues, variance explained, and PCA loadings to either a single
        file or separate files.

        Parameters
        ----------
        filename : str
            Output filename or path (without extension for CSV).
        format : str, default 'csv'
            Output format: 'csv' or 'xlsx'.
        single : bool, default True
            If True, writes all four tables into a single file (CSV:
            sequentially separated by blank lines; xlsx: one sheet at
            successive row offsets). If False, writes separate files/sheets.
        dp : int, default 3
            Decimal places for rounding.
        warm_corr : bool, default True
            Warm bias correction for person location estimates.
        tolerance : float, default 0.00001
            Newton-Raphson convergence tolerance.
        max_iters : int, default 100
            Maximum Newton-Raphson iterations.
        ext_score_adjustment : float, default 0.5
            Extreme score adjustment.
        method : str, default 'log-lik'
            Priority vector extraction method.
        constant : float, default 0.1
            Additive smoothing constant.

        Returns
        -------
        None
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
            )

        frames = [
            self.eigenvectors,
            self.eigenvalues,
            self.variance_explained,
            self.loadings,
        ]
        sheet_names_single = "Item residual analysis"
        sheet_names_multi = [
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
            with pd.ExcelWriter(
                filename, engine="openpyxl"
            ) as writer:  # BUG FIX: writer.save()
                if single:
                    row = 0
                    for frame in frames:
                        frame.round(dp).to_excel(
                            writer,
                            sheet_name=sheet_names_single,
                            startrow=row,
                            startcol=0,
                        )
                        row += frame.shape[0] + 2
                else:
                    for frame, sheet in zip(frames, sheet_names_multi):
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

    def class_intervals(self, person_locations, items=None, no_of_classes=5):
        """
        Compute class interval mean person locations and observed mean scores.

        Divides persons into quantile-based person-location groups and computes
        the mean person location and mean total score for each group. Used
        internally by ICC and TCC plot methods to overlay observed data.

        Parameters
        ----------
        person_locations : pandas.Series
            Person location estimates, indexed by person name.
        items : list of str or None, default None
            Item subset to use. None uses all items.
        no_of_classes : int, default 5
            Number of class intervals (quantile groups).

        Returns
        -------
        mean_person_locations : pandas.Series
            Mean person location estimate within each class interval.
        obs : pandas.Series
            Mean total observed score within each class interval.
        """
        class_groups = [f"class_{i + 1}" for i in range(no_of_classes)]
        if items is None:
            items = self.responses.columns.tolist()
        df = self.responses[items].dropna(how="all")
        estimates = person_locations.loc[df.index]

        quantiles = estimates.quantile(
            [(i + 1) / no_of_classes for i in range(no_of_classes - 1)]
        )
        mask_dict = {
            "class_1": estimates < quantiles.values[0],
            f"class_{no_of_classes}": estimates >= quantiles.values[-1],
            **{
                f"class_{i + 2}": (
                    (estimates >= quantiles.values[i])
                    & (estimates < quantiles.values[i + 1])
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

    def class_intervals_cats(self, item, no_of_classes=5):
        """
        Compute class interval mean person locations and observed category proportions.

        Divides persons into quantile-based person-location groups for a single item
        and computes the mean person location and the proportion of responses in each
        response category for each group. Used internally by CRC plot methods.

        Parameters
        ----------
        item : str
            Item identifier.
        no_of_classes : int, default 5
            Number of class intervals (quantile groups).

        Returns
        -------
        mean_person_locations : pandas.Series
            Mean person location estimate within each class interval.
        obs_props : numpy.ndarray
            Shape (no_of_classes, max_score_vector[item] + 1). Proportion
            of responses in each category for each class interval.
        """
        class_groups = [f"class_{i + 1}" for i in range(no_of_classes)]
        df = self.responses[item].dropna()
        estimates = self.persons[df.index]
        quantiles = estimates.quantile(
            [(i + 1) / no_of_classes for i in range(no_of_classes - 1)]
        )
        mask_dict = {
            "class_1": estimates < quantiles.values[0],
            f"class_{no_of_classes}": estimates >= quantiles.values[-1],
            **{
                f"class_{i + 2}": (
                    (estimates >= quantiles.values[i])
                    & (estimates < quantiles.values[i + 1])
                )
                for i in range(no_of_classes - 2)
            },
        }
        mean_person_locations = pd.Series(
            {cg: estimates[mask_dict[cg]].mean() for cg in class_groups}
        )
        obs_props = np.array(
            [
                [
                    (df[mask_dict[cg]] == cat).sum()
                    for cat in range(self.max_score_vector[item] + 1)
                ]
                for cg in class_groups
            ],
            dtype=float,
        )
        obs_props /= obs_props.sum(axis=1, keepdims=True)
        return mean_person_locations, obs_props

    def class_intervals_thresholds(self, item, no_of_classes=5):
        """
        Compute class interval mean person locations and observed threshold proportions.

        For each threshold of an item, conditions on persons scoring in the
        two adjacent categories, divides them into quantile-based person-location
        groups, and computes mean person location and observed proportion of the
        higher category in each group. Used internally by threshold CCS plots.

        Parameters
        ----------
        item : str
            Item identifier.
        no_of_classes : int, default 5
            Number of class intervals (quantile groups) per threshold.

        Returns
        -------
        mean_person_locations : numpy.ndarray
            Shape (no_of_classes, max_score_vector[item]). Mean person location
            within each class interval for each threshold.
        obs_props : numpy.ndarray
            Shape (no_of_classes, max_score_vector[item]). Observed proportion
            choosing the higher category in each class interval for each threshold.
        """
        if not hasattr(self, "persons"):
            self.person_estimates(warm_corr=False)

        class_groups = [f"class_{i + 1}" for i in range(no_of_classes)]
        df = self.responses[item]
        estimates = self.persons

        def make_masks(person_locations_subset):
            q = person_locations_subset.quantile(
                [(i + 1) / no_of_classes for i in range(no_of_classes - 1)]
            )
            return {
                "class_1": person_locations_subset < q.values[0],
                f"class_{no_of_classes}": person_locations_subset >= q.values[-1],
                **{
                    f"class_{i + 2}": (
                        (person_locations_subset >= q.values[i]) & (person_locations_subset < q.values[i + 1])
                    )
                    for i in range(no_of_classes - 2)
                },
            }

        mean_person_locations, obs_props = [], []
        for t in range(self.max_score_vector[item]):
            mask = df.isin([t, t + 1])
            cond_df = df[mask]
            adj_estimates = estimates[mask]
            masks = make_masks(adj_estimates)
            combined = pd.DataFrame({"estimate": adj_estimates, "score": cond_df})
            mean_person_locations.append(
                [combined["estimate"][masks[cg]].mean() for cg in class_groups]
            )
            obs_props.append(
                [(combined["score"][masks[cg]] - t).mean() for cg in class_groups]
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
        x_min=-5,
        x_max=5,
        y_max=0,
        items=None,
        curve_labels=None,
        legend_loc=None,
        max_score_lines=None,
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
        Core plotting engine for all PCM item and test characteristic curves.

        Renders one or more curves against a person-location x-axis with optional
        observed-data overlays, threshold lines, central difference lines,
        score lines, information lines, and CSEM lines. Called internally
        by icc(), crcs(), threshold_ccs(), iic(), tcc(), test_info(), and
        test_csem(). Not normally called directly by users.

        Parameters
        ----------
        x_data : array-like
            X-axis values (typically person-location grid from -20 to 20).
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
        curve_labels : list of str, or None, default None
            One label per curve (matching y_data's columns), shown in a
            legend. None (default) leaves curves unlabelled, as before.
            icc() passes item names here when plotting more than one
            item at once.
        max_score_lines : list of float, or None, default None
            One value per curve (matching y_data's columns) -- draws a
            dotted horizontal line at that value, in the same colour as
            its curve. icc() uses this to mark each item's own max score
            when items have different ones and are plotted together (so
            the shared y-axis, scaled to the tallest item, doesn't leave
            shorter items' own ceilings ambiguous).
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
            already line up.
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
            Draw vertical lines at each uncentred threshold.
        central_location : bool, default False
            Draw a vertical line at the item's central location.
        score_lines_item : list, default [None, None]
            [item_name, list_of_scores] for item-level score lines.
        score_lines_test : list or None
            Raw total scores for test-level score reference lines.
        point_info_lines_item : list, default [None, None]
            [item_name, list_of_person_locations] for item-level information lines.
        point_info_lines_test : list or None
            Person locations for test-level information reference lines.
        point_csem_lines : list or None
            Person locations for CSEM reference lines.
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
                if max_score_lines is not None:
                    ax.axhline(max_score_lines[i], color=col, linestyle=":", linewidth=1)

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
                        no_obs_cats = y_obs_data.shape[1]
                        for j in range(no_obs_cats):
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

            if items is not None:
                if isinstance(items, str) and items not in ("all", "none"):
                    thresholds = {items: self.thresholds_uncentred.loc[items].dropna()}
                elif isinstance(items, str):
                    thresholds = self.thresholds_uncentred
                else:
                    thresholds = self.thresholds_uncentred
            else:
                thresholds = self.thresholds_uncentred

            if (
                thresh_lines
                and isinstance(items, str)
                and items not in ("all", "none", None)
            ):
                item_thresholds = self.thresholds_uncentred.loc[items].dropna()
                for idx, (thr_num, thr) in enumerate(item_thresholds.items()):
                    ax.axvline(x=thr, color="black", linestyle="--")
                    # Staggered onto two rows near the bottom so that any
                    # two adjacent thresholds (the only ones close enough
                    # in x to clash) never share a row.
                    label_y = y_max * (0.05 if idx % 2 == 0 else 0.11)
                    ax.text(
                        thr + (x_max - x_min) / 100,
                        label_y,
                        str(thr_num),
                        color="black",
                        va="bottom",
                        ha="left",
                    )

            if items is not None and central_location:
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
                    if it in (None, "all", "none"):
                        continue
                    xval = np.mean(list(self.thresholds_uncentred.loc[it].dropna()))
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
                        # its ceiling by the time it nears max score, so
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
                valid = all(s > 0 for s in score_lines_item[1]) and all(
                    s < self.max_score_vector[it]
                    for it in items_list
                    for s in score_lines_item[1]
                )
                if valid:
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
                        "Invalid score for score line: values must be "
                        "strictly between 0 and the item maximum score.",
                        UserWarning,
                        stacklevel=2,
                    )

            if score_lines_test is not None:
                max_score = (
                    sum(self.max_score_vector)
                    if items is None
                    else sum(self.max_score_vector[items])
                )
                if all(s > 0 for s in score_lines_test) and all(
                    s < max_score for s in score_lines_test
                ):
                    item_keys = self.responses.columns if items is None else items
                    for s in score_lines_test:
                        estimate = self.score_lookup(
                            s, items=list(item_keys), warm_corr=False
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
                        "Invalid score for score line: values must be "
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
                        info = self.variance_uncentred(
                            estimate, self.thresholds_uncentred.loc[it].dropna()
                        )
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
                        self.variance_uncentred(
                            estimate, self.thresholds_uncentred.loc[it].dropna()
                        )
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
                        self.variance_uncentred(
                            estimate, self.thresholds_uncentred.loc[it].dropna()
                        )
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

            if cat_highlight is not None and isinstance(items, str):
                item = items
                if cat_highlight in range(self.max_score_vector[item] + 1):
                    if cat_highlight == 0:
                        ax.axvspan(
                            -100,
                            self.thresholds_uncentred.loc[item].dropna().iloc[0],
                            facecolor="blue",
                            alpha=0.2,
                        )
                    elif cat_highlight == self.max_score_vector[item]:
                        ax.axvspan(
                            self.thresholds_uncentred.loc[item].dropna().iloc[-1],
                            100,
                            facecolor="blue",
                            alpha=0.2,
                        )
                    else:
                        lo = (
                            self.thresholds_uncentred.loc[item]
                            .dropna()
                            .iloc[cat_highlight - 1]
                        )
                        hi = (
                            self.thresholds_uncentred.loc[item]
                            .dropna()
                            .iloc[cat_highlight]
                        )
                        if hi > lo:
                            ax.axvspan(lo, hi, facecolor="blue", alpha=0.2)

            if y_max <= 0:
                y_max = float(y_data.max()) * 1.1

            ax.set_xlim(x_min, x_max)
            ax.set_ylim(0, y_max)
            ax.set_xlabel(
                "Person location", fontsize=axis_font_size, fontweight="bold", wrap=True
            )
            ax.set_ylabel(
                y_label, fontsize=axis_font_size, fontweight="bold", wrap=True
            )
            ax.set_title(
                graph_title, fontsize=title_font_size, fontweight="bold", wrap=True
            )
            ax.grid(True)
            ax.tick_params(axis="x", labelsize=labelsize)
            ax.tick_params(axis="y", labelsize=labelsize)

            if filename is not None:
                graph.savefig(filename + f".{file_format}", dpi=plot_density)

            # Close the figure before returning. In Jupyter, two display events
            # fire if the figure is still open when returned: one from show()
            # and one from the notebook's auto-display of the returned object.
            # Closing here prevents the second. The figure data is preserved in
            # the returned object so callers can still inspect axes, save etc.
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

        Displays modelled expected score as a function of person location.
        Optionally overlays observed class-interval mean scores -- single-item
        only, see item below.

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
            so they stay distinguishable. Items can have different
            max_score_vector values; the y-axis is scaled to the largest
            among the plotted items.
        obs : bool, default False
            If True, overlays observed class-interval mean scores. Works
            with a list of items too -- one column of observed points
            per item, in that item's own curve colour (class intervals
            are the same person-location quantile groups for every item,
            computed from self.persons, so the same x-axis positions are
            shared).
        no_of_classes : int, default 5
            Number of class intervals for the observed overlay.
        title : str or None, default None
            Plot title.
        thresh_lines : bool, default False
            Draw vertical lines at each threshold. Single item only --
            silently ignored if item is a list.
        central_location : bool, default False
            Draw a line at the central item location, labelled with the
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
        multi = isinstance(item, list)
        if multi:
            # thresh_lines/cat_highlight each only make sense for a single
            # item's own location -- silently no-op with several items
            # plotted at once rather than erroring, since they'd just be
            # visual clutter with no obvious single "right" item anyway.
            thresh_lines = False
            cat_highlight = None

        # BUG FIX: variable name typo in original; now self.persons
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
                    mpl, om = self.class_intervals(
                        items=[it], person_locations=self.persons,
                        no_of_classes=no_of_classes,
                    )
                    if xobsdata is None:
                        xobsdata = pd.Series(mpl)
                    yobs_cols.append(np.array(om))
                yobsdata = np.column_stack(yobs_cols)
            else:
                mean_person_locations, obs_means = self.class_intervals(
                    items=item, person_locations=self.persons, no_of_classes=no_of_classes
                )
                xobsdata = pd.Series(mean_person_locations)
                yobsdata = np.array(obs_means).reshape(-1, 1)

        estimates = np.arange(-20, 20, 0.1)
        if multi:
            y = np.column_stack(
                [
                    [
                        self.exp_score_uncentred(
                            a, self.thresholds_uncentred.loc[it].dropna()
                        )
                        for a in estimates
                    ]
                    for it in item
                ]
            )
            y_max = max(self.max_score_vector[it] for it in item)
        else:
            y = np.array(
                [
                    self.exp_score_uncentred(
                        a, self.thresholds_uncentred.loc[item].dropna()
                    )
                    for a in estimates
                ]
            ).reshape(-1, 1)
            y_max = self.max_score_vector[item]

        return self.plot_data(
            x_data=estimates,
            y_data=y,
            x_obs_data=xobsdata,
            y_obs_data=yobsdata,
            x_min=xmin,
            x_max=xmax,
            y_max=y_max,
            items=item,
            curve_labels=item if multi else None,
            legend_loc="upper left" if score_lines is not None else "lower right",
            max_score_lines=(
                [self.max_score_vector[it] for it in item] if multi else None
            ),
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
            cat_highlight=cat_highlight,
            black=black,
            font=font,
            title_font_size=title_font_size,
            axis_font_size=axis_font_size,
            labelsize=labelsize,
            filename=filename,
            plot_density=dpi,
            file_format=file_format,
        )

    def crcs(
        self,
        item,
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
        person location using uncentred PCM parameterisation. Optionally overlays
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
        central_location : bool, default False
            Draw a line at the central location.
        cat_highlight : int or None, default None
            Category to shade.
        xmin, xmax : float
            Person-location axis limits.
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
        # BUG FIX: variable name typo in original (now self.persons)
        if obs is not None and not hasattr(self, "persons"):
            self.person_estimates(warm_corr=False)
        if item == "none":
            item = None

        xobsdata = yobsdata = np.array(np.nan)
        obs_curve_index = None
        if obs is not None:
            mean_person_locations, obs_props = self.class_intervals_cats(
                item=item, no_of_classes=no_of_classes
            )
            xobsdata, yobsdata = mean_person_locations, obs_props
            if obs != "all":
                if not all(
                    c in np.arange(self.max_score_vector[item] + 1) for c in obs
                ):
                    warnings.warn(
                        "Invalid 'obs' value. Valid values are None, 'all', "
                        "or a list of category indices.",
                        UserWarning,
                        stacklevel=2,
                    )
                    return
                yobsdata = yobsdata[:, obs]
                obs_curve_index = list(obs)

        estimates = np.arange(-20, 20, 0.1)
        y = np.array(
            [
                [
                    self.cat_prob_uncentred(
                        a, cat, self.thresholds_uncentred.loc[item].dropna()
                    )
                    for cat in range(self.max_score_vector[item] + 1)
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
            curve_labels=[
                f"Category {c}" for c in range(self.max_score_vector[item] + 1)
            ],
            graph_title=title or "",
            y_label="Probability",
            obs=obs,
            obs_curve_index=obs_curve_index,
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
        item,
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
        Plot Threshold Characteristic Curves (TCCs) for a single item.

        Displays the probability of scoring in the higher of two adjacent
        categories at each threshold, as a function of person location.

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
        central_location : bool, default False
            Draw a line at the central location.
        cat_highlight : int or None, default None
            Threshold category to shade.
        xmin, xmax : float
            Person-location axis limits.
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
        if obs is not None and not hasattr(self, "persons"):
            self.person_estimates(warm_corr=False)

        xobsdata = yobsdata = np.array(np.nan)
        obs_curve_index = None
        if obs is not None:
            mean_person_locations, obs_props = self.class_intervals_thresholds(
                item, no_of_classes=no_of_classes
            )
            xobsdata, yobsdata = mean_person_locations, obs_props
            if obs != "all":
                if not all(
                    c in np.arange(self.max_score_vector[item]) + 1 for c in obs
                ):
                    warnings.warn(
                        "Invalid 'obs' value. Valid values are None, 'all', "
                        "or a list of threshold indices.",
                        UserWarning,
                        stacklevel=2,
                    )
                    return
                obs_idx = [o - 1 for o in obs]
                xobsdata = xobsdata[:, obs_idx]
                yobsdata = yobsdata[:, obs_idx]
                obs_curve_index = obs_idx

        estimates = np.arange(-20, 20, 0.1)
        item_thresholds = self.thresholds_uncentred.loc[item].dropna()
        y = np.array(
            [
                [1.0 / (1.0 + np.exp(thr - a)) for thr in item_thresholds]
                for a in estimates
            ]
        )

        return self.plot_data(
            x_data=estimates,
            y_data=y,
            y_max=1,
            x_min=xmin,
            x_max=xmax,
            items=item,
            curve_labels=[f"Threshold {t}" for t in item_thresholds.index],
            x_obs_data=xobsdata,
            y_obs_data=yobsdata,
            graph_title=title or "",
            y_label="Probability",
            obs=obs,
            obs_curve_index=obs_curve_index,
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

        Displays Fisher information (item variance) as a function of person location
        using uncentred threshold parameterisation.

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
            Draw vertical lines at each threshold. Single item only --
            silently ignored if item is a list.
        central_location : bool, default False
            Draw a line at the central location. Works with a list of
            items too.
        point_info_lines : list or None, default None
            Person location values at which to draw information reference lines.
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
                    [
                        self.variance_uncentred(
                            a, self.thresholds_uncentred.loc[it].dropna()
                        )
                        for a in estimates
                    ]
                    for it in item
                ]
            )
        else:
            y = np.array(
                [
                    self.variance_uncentred(a, self.thresholds_uncentred.loc[item].dropna())
                    for a in estimates
                ]
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
            thresh_lines=thresh_lines,
            items=item,
            curve_labels=item if multi else None,
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
            file_format=file_format,
            plot_density=dpi,
        )

    def tcc(
        self,
        items=None,
        obs=False,
        xmin=-5,
        xmax=5,
        no_of_classes=5,
        title=None,
        score_lines=None,
        score_labels=False,
        warm=True,
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
        xmin, xmax : float
            Person-location axis limits.
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
        if isinstance(items, str) and items in ("all", "none"):
            items = None
        elif isinstance(items, str):
            items = [items]

        # BUG FIX: variable name typo in original (now self.persons)
        if obs and not hasattr(self, "persons"):
            self.person_estimates(warm_corr=False)

        xobsdata = yobsdata = np.array(np.nan)
        if obs:
            mean_person_locations, obs_means = self.class_intervals(
                items=items, person_locations=self.persons, no_of_classes=no_of_classes
            )
            xobsdata = mean_person_locations
            yobsdata = np.array(obs_means).reshape(no_of_classes, 1)

        estimates = np.arange(-20, 20, 0.1)
        item_keys = self.responses.columns if items is None else items
        y = np.array(
            [
                sum(
                    self.exp_score_uncentred(
                        a, self.thresholds_uncentred.loc[it].dropna()
                    )
                    for it in item_keys
                )
                for a in estimates
            ]
        ).reshape(-1, 1)
        y_max = sum(self.max_score_vector[it] for it in item_keys)

        return self.plot_data(
            x_data=estimates,
            y_data=y,
            x_obs_data=xobsdata,
            y_obs_data=yobsdata,
            x_min=xmin,
            x_max=xmax,
            y_max=y_max,
            items=items,
            score_lines_test=score_lines,
            score_labels=score_labels,
            warm=warm,
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
            file_format=file_format,
            plot_density=dpi,
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
            Person location values at which to draw reference lines.
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
        if isinstance(items, str) and items in ("all", "none"):
            items = None
        elif isinstance(items, str):
            items = [items]
        item_keys = self.responses.columns if items is None else items
        estimates = np.arange(-20, 20, 0.1)
        y = np.array(
            [
                sum(
                    self.variance_uncentred(
                        a, self.thresholds_uncentred.loc[it].dropna()
                    )
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
            x_min=xmin,
            x_max=xmax,
            y_max=ymax,
            items=items,
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
            file_format=file_format,
            plot_density=dpi,
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

        Displays 1 / sqrt(I(theta)) as a function of person location, where I(theta)
        is total test information.

        Parameters
        ----------
        items : str, list, or None, default None
            Item subset. None uses all items.
        point_csem_lines : list or None, default None
            Person location values at which to draw CSEM reference lines.
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
        if isinstance(items, str) and items in ("all", "none"):
            items = None
        elif isinstance(items, str):
            items = [items]
        item_keys = self.responses.columns if items is None else items
        estimates = np.arange(-20, 20, 0.1)
        info = np.array(
            [
                sum(
                    self.variance_uncentred(
                        a, self.thresholds_uncentred.loc[it].dropna()
                    )
                    for it in item_keys
                )
                for a in estimates
            ]
        )
        y = (1.0 / (info**0.5)).reshape(-1, 1)

        return self.plot_data(
            x_data=estimates,
            y_data=y,
            x_min=xmin,
            x_max=xmax,
            y_max=ymax,
            items=items,
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
            file_format=file_format,
            plot_density=dpi,
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
        black : bool, default False
            If True, renders the histogram in black.
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
            Deliberately breaks the "same Count/Density value reads at
            the same physical scale" property that item_distribution's
            panel sizing otherwise guarantees, in favour of giving that
            panel more usable room -- useful when there are far more
            persons than items, which otherwise leaves the item panel's
            own real variation looking like a flat, uninformative
            squiggle at persons' own scale. person_scaling=10 makes the
            item panel 10x larger than a strict equal-scale match would
            give it (so persons' own axis effectively reads 10x coarser
            than items') -- persons' and items' own displayed axis
            values are unaffected either way, only how much physical
            space item_distribution's panel gets. Only has an effect
            when item_distribution=True (the only case this panel-size
            matching applies to for RSM/PCM).
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
                    item_weights = item_sign * np.ones_like(items)
                    if prop:
                        item_weights = item_weights / no_of_items

                    item_ax.hist(
                        items,
                        weights=item_weights,
                        bins=bins,
                        color=item_color,
                        edgecolor=edge_color,
                        label="Items",
                        alpha=alpha,
                        orientation=orientation,
                    )

                else:
                    kde_items = gaussian_kde(items, bw_method=bw_method)(x_grid)
                    item_curve = item_sign * (
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
                            label="Items",
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
                            label="Items",
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
                ax_lo = -(person_range if person_sign < 0 else item_range)
                ax_hi = person_range if person_sign > 0 else item_range
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

            def _relabel(ax_obj, pos_reach, pos_step, neg_reach, neg_step, drop_zero=False):
                ticks = sorted(
                    set(_side_ticks(pos_reach, pos_step))
                    | {-t for t in _side_ticks(neg_reach, neg_step)}
                )
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
                    ticks = [t for t in ticks if t != 0]
                labels = (
                    [f"{abs(t):.2f}" for t in ticks]
                    if prop
                    else [str(int(round(abs(t)))) for t in ticks]
                )
                if is_vertical:
                    ax_obj.set_yticks(ticks)
                    ax_obj.set_yticklabels(labels, fontsize=labelsize)
                else:
                    ax_obj.set_xticks(ticks)
                    ax_obj.set_xticklabels(labels, fontsize=labelsize)

            if item_labels:
                if person_sign > 0:
                    _relabel(ax, person_range, person_step, 0, 1)
                else:
                    _relabel(ax, 0, 1, person_range, person_step)
            elif person_sign > 0:
                _relabel(ax, person_range, person_step, item_range, item_step)
            else:
                _relabel(ax, item_range, item_step, person_range, person_step)

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
