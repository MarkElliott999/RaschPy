from math import log
import warnings
from scipy.stats import chi2, norm

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from matplotlib import pyplot as plt
from matplotlib import colors as colors
from matplotlib import cm as cmx
import seaborn as sns
import matplotlib.patheffects as path_effects

from raschpy.base import Rasch, _BootstrapWarnings, _save_table


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

        self.model = "PCM"

        # Sim-aware instantiation: store sim attributes in self.generating namespace
        from raschpy.simulation.pcm_sim import PCM_Sim
        from raschpy.simulation.slm_sim import SLM_Sim
        from raschpy.simulation.rsm_sim import RSM_Sim
        from raschpy.base import _SimParams

        if isinstance(responses, (SLM_Sim, RSM_Sim)):
            # Cross-model construction -- lets recovery_stats() compare a
            # PCM fit against another model class's generating data (e.g.
            # for a person-recovery pseudo-estimate check, or an item/
            # threshold nesting check when the pairing is genuinely
            # comparable). PCM has no single max_score_vector to infer
            # from these sim classes, so it must be given explicitly.
            sim = responses
            self.generating = _SimParams()
            for attr, value in vars(sim).items():
                setattr(self.generating, attr, value)
            if max_score_vector is None:
                raise ValueError(
                    f"max_score_vector must be given explicitly when "
                    f"constructing PCM from a {type(sim).__name__} (PCM "
                    f"has no single max_score_vector to infer from one)."
                )
            self.max_score_vector = pd.Series(
                {item: int(ms) for item, ms in zip(sim.item_names, max_score_vector)}
            )
            responses = sim.responses
        elif isinstance(responses, PCM_Sim):
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

        # Unobserved score categories (bottom, middle or top). The thresholds
        # next to an unobserved category are not identified by the data, so
        # their estimates are set by the smoothing constant.
        unobserved = {}
        for item in self.max_score_vector.index:
            seen = set(responses[item].dropna().astype(int).unique())
            missing_cats = [
                k for k in range(int(self.max_score_vector[item]) + 1) if k not in seen
            ]
            if missing_cats:
                unobserved[item] = missing_cats
        if unobserved:
            cells = [
                f"item {item}, category {k}: 0 responses"
                for item, cats in unobserved.items()
                for k in cats
            ]
            shown = "; ".join(cells[:20])
            if len(cells) > 20:
                shown += f"; and {len(cells) - 20} more"
            warnings.warn(
                f"Unused score categories ({shown}). The thresholds next to these categories are not "
                f"identified by the data, so their estimates are determined by the "
                f"smoothing constant rather than the responses. Consider collapsing "
                f"unused categories (e.g. reducing that item's max_score_vector entry if "
                f"its top or bottom category is unused) or collecting more data.",
                UserWarning,
                stacklevel=2,
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

            # check_data_connectivity() itself warns about disconnection,
            # structural zeros and weak connectivity (one warning per problem).

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

    def _cell_constants(self, n_obs, miss_obs):
        """Per-cell smoothing constants for calibrate(constant=None): cell
        (a, b) of the threshold-pair matrix gets base * (K_a * K_b) ** -0.7,
        base being the data-adaptive rule at K = 1; reduces to the scalar
        rule when all items share one K."""
        row_k = np.repeat(
            self.max_score_vector.to_numpy(dtype=float),
            self.max_score_vector.to_numpy(dtype=int),
        )
        scale = row_k**-0.7
        return self._default_constant(n_obs, miss_obs, 1.0) * np.outer(scale, scale)

    def _unused_category_rows(self, row_items):
        """Rows (item i, threshold k) of the item-category pairwise matrix
        next to an unused category of item i: row (i, k) has no wins when
        nobody scored k, and no losses when nobody scored k - 1."""
        x = self.responses.to_numpy(dtype=float)
        mask = []
        for i, item in enumerate(self.item_names):
            col = x[:, i]
            for k in range(1, int(self.max_score_vector[item]) + 1):
                mask.append((col == k).sum() == 0 or (col == k - 1).sum() == 0)
        return np.array(mask, dtype=bool)

    def _pairwise_harmonic_weights(self):
        """
        Per (item, threshold) harmonic-mean information weight, from the raw
        (unsmoothed, unpowered) pairwise comparison matrix -- row-summed
        element-wise harmonic mean of the matrix and its transpose:
        sum_j 2*wins_ij*losses_ij / (wins_ij+losses_ij).

        For a Bradley-Terry pair this is proportional to n_ij*p_ij*(1-p_ij),
        that pair's Fisher-information contribution -- the same harmonic-mean
        information-weighting idea behind RSM's CPAT total_weight (Elliott &
        Buttery, 2022), expressed in PCM's own cross-item comparison
        structure. Confirmed empirically as the tightest available SE proxy
        for PCM's per-threshold precision (see raschpy_dynamic_n_calibration
        project notes) -- used as the effective sample size for dif_test's
        DCF dynamic SE cap (dcf_cap).

        Returns
        -------
        pandas.Series
            Harmonic weight per item-threshold, indexed by a (Item,
            Category) MultiIndex matching threshold_dif_table's own index.
        """
        matrix, row_items = self._build_pairwise_matrix()
        with np.errstate(divide="ignore", invalid="ignore"):
            denom = matrix + matrix.T
            safe_denom = np.where(denom > 0, denom, 1.0)
            hmean = np.where(denom > 0, 2.0 * matrix * matrix.T / safe_denom, 0.0)
        np.fill_diagonal(hmean, 0.0)
        weights = hmean.sum(axis=1)

        categories = []
        running = {}
        for item in row_items:
            running[item] = running.get(item, 0) + 1
            categories.append(running[item])
        index = pd.MultiIndex.from_arrays(
            [row_items, categories], names=["Item", "Category"]
        )
        return pd.Series(weights, index=index)

    @staticmethod
    def _default_constant(no_of_persons, missing, mean_max_score):
        """
        Data-adaptive default smoothing constant for calibrate(constant=None):
        0.1 * N ** 0.15 * (1 - missing) ** 0.75 * K ** -1.4, with N the number
        of persons with at least one response, missing the proportion of
        missing responses among them, and K the mean maximum score. Fitted by
        simulation to keep person-measurement error from item estimation
        (estimated vs. true parameters) close to the best constant for each
        design, and validated out of sample.
        """
        n = max(no_of_persons, 1)
        m = min(max(missing, 0.0), 0.99)
        return 0.1 * n**0.15 * (1 - m) ** 0.75 * max(mean_max_score, 1) ** -1.4

    def calibrate(
        self,
        constant=None,
        method="log-lik",
        matrix_power=None,
        log_lik_tol=0.000001,
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
            ``None`` resolves to a data-adaptive value,
            0.1 * N ** 0.15 * (1 - missing) ** 0.75 * K ** -1.4, where N is the
            number of persons with at least one response, missing is the
            proportion of missing responses among them, and K is the mean
            maximum score across items. With mixed maximum scores the
            constant is applied per cell of the threshold-pair matrix:
            cell (a, b) gets 0.1 * N ** 0.15 * (1 - missing) ** 0.75 *
            (K_a * K_b) ** -0.7, which reduces to the formula above when all
            items share one maximum score. Use 0 to disable smoothing;
            estimation may fail if any item has all-maximum or all-minimum
            scores.
        method : str, default 'log-lik'
            Priority vector extraction method. See base.priority_vector().
        matrix_power : int or None, default None
            Initial matrix power before checking for structural zeroes.
            None resolves to 0 for method='log-lik' (no powering), else 5.
        log_lik_tol : float, default 0.000001
            Log-likelihood convergence tolerance for priority vector extraction.
        robust : bool, default False
            Deprecated and has no effect; passing ``robust=True`` raises a
            ``FutureWarning``. The two-pass rule it selected is outperformed
            by the data-adaptive ``constant=None`` default, including in the
            sparse, many-category designs it was built for. It will be
            removed in a future release.

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
            ``None``. With mixed maximum scores under ``constant=None``, the
            equal-K equivalent using the mean maximum score.
        cell_constants : numpy.ndarray or None
            Per-cell constants applied to the threshold-pair matrix when
            ``constant=None``; ``None`` when an explicit constant is given.
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

        if robust:
            warnings.warn(
                "`robust` is deprecated and has no effect: the default "
                "`constant=None` is now data-adaptive, "
                "0.1 * N ** 0.15 * (1 - missing) ** 0.75 * K ** -1.4. The argument "
                "will be removed in a future release.",
                FutureWarning,
                stacklevel=2,
            )

        cell_constants = None
        if constant is None:
            observed = self.responses[self.responses.notna().any(axis=1)]
            n_obs = len(observed)
            miss_obs = float(observed.isna().to_numpy().mean()) if n_obs else 0.0
            # Scalar equal-K equivalent (recorded as self.constant), and the
            # per-cell constants actually applied: cell (a, b) of the
            # threshold-pair matrix gets base * (K_a * K_b) ** -0.7, which
            # reduces to the scalar rule when all items share one K.
            resolved_constant = self._default_constant(
                n_obs, miss_obs, float(self.max_score_vector.mean())
            )
            cell_constants = self._cell_constants(n_obs, miss_obs)
        else:
            resolved_constant = constant
        if resolved_constant == 0:
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

        constant = resolved_constant
        matrix_power = self._resolve_matrix_power(method, matrix_power)

        C = cell_constants if cell_constants is not None else constant
        constant_matrix = ((matrix + matrix.T) > 0).astype(np.float64) * C
        matrix = matrix + constant_matrix
        np.fill_diagonal(
            matrix,
            matrix.diagonal()
            + (np.diag(C) if cell_constants is not None else constant),
        )

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
                    fill = off_diagonal_mask & (mat == 0)
                    mat[fill] = C[fill] if cell_constants is not None else constant
            else:
                mat = np.linalg.matrix_power(matrix, matrix_power)
                mat_pow = matrix_power
                while 0 in mat:
                    mat = mat @ matrix
                    mat_pow += 1
                    if mat_pow == matrix_power + 5:
                        mat += C
                        break

        threshold_vector = self.priority_vector(
            mat, method=method, log_lik_tol=log_lik_tol, pcm=True
        )
        self.threshold_list = threshold_vector
        self.constant = resolved_constant
        self.cell_constants = cell_constants

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
        self.thresholds_uncentred.columns = range(
            1, self.thresholds_uncentred.shape[1] + 1
        )

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

        self._calibrate_anchor_core(
            anchors,
            selection_method,
            corr_tol,
            sd_ratio_tol,
            min_anchors,
            wald_alpha,
            no_of_samples,
            adj,
            overwrite_anchors,
            plot,
            plot_kwargs,
            seed,
            std_errors_kwargs=dict(
                constant=constant,
                robust=robust,
                method=method,
                matrix_power=matrix_power,
                log_lik_tol=log_lik_tol,
            ),
        )

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

        self._snapshot_calibrate_anchor_run(
            anchors,
            extra=dict(
                anchor_threshold_test=self.anchor_threshold_test,
                anchor_threshold_test_aggregate=self.anchor_threshold_test_aggregate,
            ),
        )

    def _threshold_structure_ll(
        self,
        thresholds_uncentred,
        warm_corr=True,
        tolerance=0.00001,
        max_iters=100,
        ext_score_adjustment=0.5,
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

    def _sandwich_covariance(self, small_sample=False):
        """
        Sandwich covariance of the uncentred thresholds (cells in
        _build_pairwise_matrix() order) for the Bradley-Terry estimating
        equations of PAIR: person n contributes a 'win' for cell (i, k) over
        cell (j, m) when scoring k on item i and m - 1 on item j (i != j),
        and the strength of cell (i, k) is minus its uncentred threshold.
        Persons are the independent units; the bread's one-dimensional null
        space (location origin) is handled with a pseudo-inverse.
        small_sample=True applies the Kauermann-Carroll correction to the
        scores (see _kc_scores()).
        """
        x = self.responses.to_numpy(dtype=float)
        rows, cols, deltas = [], [], []
        for i, item in enumerate(self.item_names):
            max_k = int(self.max_score_vector[item])
            onehot = np.zeros((x.shape[0], max_k + 1))
            valid = ~np.isnan(x[:, i])
            onehot[valid, x[valid, i].astype(int)] = 1.0
            rows.append(onehot[:, 1:])
            cols.append(onehot[:, :max_k])
            deltas.append(
                self.thresholds_uncentred.loc[item].dropna().values.astype(float)
            )
        R, C = np.hstack(rows), np.hstack(cols)
        b = -np.concatenate(deltas)
        owner = np.repeat(
            np.arange(len(self.item_names)),
            [int(self.max_score_vector[it]) for it in self.item_names],
        )
        mask = (owner[:, None] != owner[None, :]).astype(float)
        with np.errstate(over="ignore"):
            pi = 1.0 / (1.0 + np.exp(-(b[:, None] - b[None, :])))
        Pm = pi * mask
        psi = R * (C @ mask.T) - R * (C @ Pm.T) - C * (R @ Pm.T)
        wins = R.T @ C
        w = (wins + wins.T) * mask * pi * (1 - pi)
        L = np.diag(w.sum(axis=1)) - w
        Lp = self._origin_pinv(L)
        if small_sample:
            v = mask * pi * (1 - pi)
            psi = self._kc_scores(
                psi, lambda sl: self._bt_person_curvature([(R[sl], C[sl])], v), Lp
            )
        return Lp @ (psi.T @ psi) @ Lp, owner

    def _central_covariance(self):
        """Sandwich covariance of the central item locations (each the mean of
        its item's uncentred thresholds), from self.se_covariance."""
        V = self.se_covariance.values
        names = list(self.item_names)
        owner = np.repeat(
            np.arange(len(names)), [int(self.max_score_vector[it]) for it in names]
        )
        A = np.zeros((len(names), V.shape[0]))
        for i in range(len(names)):
            idx = np.flatnonzero(owner == i)
            A[i, idx] = 1.0 / len(idx)
        return pd.DataFrame(A @ V @ A.T, index=names, columns=names)

    def _sandwich_std_errors(self, interval):
        """Set the std_errors() attributes from the sandwich covariance."""
        V, owner = self._sandwich_covariance()
        names = list(self.item_names)
        labels = [
            (it, k)
            for it in names
            for k in range(1, int(self.max_score_vector[it]) + 1)
        ]
        self.se_covariance = pd.DataFrame(
            V,
            index=pd.MultiIndex.from_tuples(labels),
            columns=pd.MultiIndex.from_tuples(labels),
        )
        z = norm.ppf(0.5 + interval / 2) if interval is not None else None
        self.threshold_se, self.cat_width_se, central_se = {}, {}, {}
        lows = {"thr": {}, "cw": {}, "c": {}}
        highs = {"thr": {}, "cw": {}, "c": {}}
        for i, item in enumerate(names):
            idx = np.flatnonzero(owner == i)
            Vi = V[np.ix_(idx, idx)]
            K = len(idx)
            thr = self.thresholds_uncentred.loc[item].dropna().values.astype(float)
            thr_se = np.sqrt(np.clip(np.diag(Vi), 0, None))
            a = np.full(K, 1.0 / K)  # central location = mean threshold
            c_se = float(np.sqrt(max(a @ Vi @ a, 0.0)))
            if K > 1:
                Dm = np.diff(np.eye(K), axis=0)  # widths = consecutive differences
                cw_se = np.sqrt(np.clip(np.diag(Dm @ Vi @ Dm.T), 0, None))
                widths = np.diff(thr)
            else:
                cw_se, widths = np.zeros(1), np.zeros(1)
            self.threshold_se[item] = thr_se
            self.cat_width_se[item] = pd.Series(cw_se, index=range(1, len(cw_se) + 1))
            central_se[item] = c_se
            if z is not None:
                lows["thr"][item], highs["thr"][item] = (
                    thr - z * thr_se,
                    thr + z * thr_se,
                )
                lows["cw"][item], highs["cw"][item] = (
                    widths - z * cw_se,
                    widths + z * cw_se,
                )
                lows["c"][item] = self.items[item] - z * c_se
                highs["c"][item] = self.items[item] + z * c_se
        self.item_se = pd.Series(central_se)
        self.central_bootstrap = self.threshold_bootstrap = self.cat_width_bootstrap = (
            None
        )
        self.bootstrap_sample_thresholds = None
        if z is not None:
            self.threshold_low, self.threshold_high = lows["thr"], highs["thr"]
            self.cat_width_low, self.cat_width_high = lows["cw"], highs["cw"]
            self.central_low, self.central_high = pd.Series(lows["c"]), pd.Series(
                highs["c"]
            )
        else:
            self.threshold_low = self.threshold_high = None
            self.cat_width_low = self.cat_width_high = None
            self.central_low = self.central_high = None

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
        se_method=None,
    ):
        """
        Estimate standard errors for thresholds, category widths and central
        item locations.

        se_method='sandwich' computes analytic standard errors: a sandwich
        (robust) covariance for the Bradley-Terry estimating equations of
        PAIR on the item-category pairwise matrix (each cell's strength is
        minus its uncentred threshold), with persons as the independent
        units. Central locations, centred thresholds and category widths
        are linear functions of the uncentred thresholds, so their
        covariances follow exactly. It costs about one calibration.

        se_method='bootstrap' draws no_of_samples bootstrap resamples of
        person-level response data, calibrates each resample, and computes
        the standard deviation of the estimates across samples.

        se_method=None (default) uses the sandwich unless some item has a
        score category that nobody used (its adjacent thresholds are then
        set by the smoothing constant, which the sandwich does not model),
        in which case it uses the bootstrap. self.se_method records the
        method used.

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
        se_method : {None, 'sandwich', 'bootstrap'}, default None
            Standard error method; see above. With 'sandwich', confidence
            intervals are estimate +/- z * SE, and the bootstrap attributes
            are None.

        Attributes set
        --------------
        se_method : str
            'sandwich' or 'bootstrap', the method used.
        se_covariance : pandas.DataFrame or None
            Sandwich covariance of the uncentred thresholds (sandwich only).
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
            {item: DataFrame} of bootstrap threshold estimates (None with
            the sandwich).
        """
        if se_method not in (None, "sandwich", "bootstrap"):
            raise ValueError(
                f"se_method must be None, 'sandwich' or 'bootstrap', got {se_method!r}."
            )
        if not hasattr(self, "thresholds_uncentred"):
            self.calibrate(
                constant=constant,
                method=method,
                robust=robust,
                matrix_power=matrix_power,
                log_lik_tol=log_lik_tol,
            )
        if se_method is None:
            x = self.responses.to_numpy(dtype=float)
            unobserved = [
                f"item {item}, category {k}: 0 responses"
                for i, item in enumerate(self.item_names)
                for k in range(int(self.max_score_vector[item]) + 1)
                if (x[:, i] == k).sum() == 0
            ]
            se_method = "bootstrap" if unobserved else "sandwich"
            if unobserved:
                shown = "; ".join(unobserved[:20])
                if len(unobserved) > 20:
                    shown += f"; and {len(unobserved) - 20} more"
                warnings.warn(
                    f"Standard errors computed by bootstrap rather than analytically "
                    f"(sandwich), because of unused score categories "
                    f"({shown}): the thresholds next to them are "
                    f"set by the smoothing constant, which the analytic standard "
                    f"errors do not account for. Consider collapsing unused "
                    f"categories or collecting more data. Pass se_method= to choose "
                    f"the method explicitly.",
                    UserWarning,
                    stacklevel=2,
                )
        self.se_method = se_method
        if se_method == "sandwich":
            self._sandwich_std_errors(interval)
            return
        self.se_covariance = None

        rng = np.random.default_rng(seed)
        samples = []
        collector = _BootstrapWarnings()
        for _ in range(no_of_samples):
            with collector.resample():
                sample = PCM(
                    self.responses.sample(frac=1, replace=True, random_state=rng),
                    self.max_score_vector,
                )
                sample.calibrate(
                    constant=constant,
                    robust=robust,
                    method=method,
                    matrix_power=matrix_power,
                    log_lik_tol=log_lik_tol,
                )
            samples.append(sample)
        collector.emit()

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
            If True, returns Warm's (1989) weighted likelihood estimate, found by
            solving the weighted likelihood equation directly (bracketed
            Newton). If False, returns the plain ML estimate.
        tolerance : float, default 0.00001
            Newton-Raphson convergence criterion.
        max_iters : int, default 100
            Maximum Newton-Raphson iterations.
        ext_score_adjustment : float, default 0.5
            Amount added to a score of 0 (or subtracted from a maximum score)
            for plain ML (warm_corr=False), which has no finite estimate for
            extreme scores; also sets the starting value. Has no effect on
            Warm estimates, which are finite for extreme scores.

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
            estimates = (
                np.log(scores) - np.log(ext_scores - scores) + mean_item_locations
            )
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
                pf_arr = person_filter.loc[estimates.index].values
                estimates = pd.Series(
                    self._solve_wle(
                        person_data.loc[estimates.index].sum(axis=1).values,
                        estimates.values,
                        lambda t, rows: self._wle_moments(t, items_list, pf_arr[rows]),
                        tolerance,
                        max_iters,
                    ),
                    index=estimates.index,
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
            pf_arr = np.ones((1, len(items)))
            estimate = float(
                self._solve_wle(
                    [float(score)],
                    [estimate],
                    lambda t, rows: self._wle_moments(t, list(items), pf_arr[rows]),
                    tolerance,
                    max_iters,
                )[0]
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
        filename=None,
        file_format="csv",
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
        filename : str or None, default None
            If given, also saves the returned table to
            ``{filename}.{file_format}`` (path without extension, as for
            the plotting methods).
        file_format : {'csv', 'xlsx'}, default 'csv'
            File format used when filename is given.

        Attributes set
        --------------
        score_table : pandas.Series
            Person location estimate for each possible raw score, indexed by score.

        Returns
        -------
        pandas.Series
            The table, also stored as the attribute listed above.
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
            pf_arr = np.ones((len(scores), len(items)))
            estimates = pd.Series(
                self._solve_wle(
                    scores,
                    estimates.values,
                    lambda t, rows: self._wle_moments(t, list(items), pf_arr[rows]),
                    tolerance,
                    max_iters,
                ),
                index=scores,
            )

        self.score_table = estimates
        _save_table(self.score_table, filename, file_format)
        return self.score_table

    def _wle_moments(self, person_locations, items, person_filter):
        """
        Expected score, information, and third and fourth cumulants of the
        total score at the given person locations, for _solve_wle().

        person_locations : array (N,); items : list of item names;
        person_filter : array (N, I), 1 = observed, 0 or NaN = missing.
        """
        probs, cats_arr = self._cat_probs_matrix(
            np.asarray(person_locations, dtype=float),
            list(items),
            self.thresholds_uncentred,
        )
        return self._polytomous_moments(probs, cats_arr, person_filter)

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

        probs, cats_arr = self._cat_probs_matrix(
            person_locations.values, items, thresholds
        )
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

        probs, cats_arr = self._cat_probs_matrix(
            person_locations.values, items, thresholds
        )
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

    def category_counts_df(
        self,
        persons=None,
        items=None,
        counts_name=None,
        filename=None,
        file_format="csv",
    ):
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
        filename : str or None, default None
            If given, also saves the returned table to
            ``{filename}.{file_format}`` (path without extension, as for
            the plotting methods).
        file_format : {'csv', 'xlsx'}, default 'csv'
            File format used when filename is given.

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

        _save_table(category_counts_df, filename, file_format)
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
        self,
        responses,
        theta,
        step_shape,
        item_names,
        tolerance=0.00001,
        max_iters=100,
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
            exp_score_df = pd.DataFrame(
                exp_score, index=theta.index, columns=active_items
            )
            exp_score_df *= person_filter[active_items]
            dev = cats_arr[:, None, None] - exp_score[None, :, :]
            info = (dev**2 * probs).sum(axis=0)
            info_df = pd.DataFrame(info, index=theta.index, columns=active_items)
            info_df *= person_filter[active_items]

            exp_sum = exp_score_df.sum(axis=0)
            info_sum = info_df.sum(axis=0)

            changes = ((exp_sum - obs_scores[active_items]) / info_sum).clip(-1, 1)
            delta.loc[active_items] += changes
            active.loc[active_items] = changes.abs() > tolerance
            iters += 1

        thresholds_final = step_shape.add(delta, axis=0)
        probs, cats_arr = self._cat_probs_matrix(
            theta.values, item_names, thresholds_final
        )
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

    def _global_fit_subset(self, idx):
        """Calibrated PCM on a subset of persons (global_fit())."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = PCM(
                self.responses.loc[idx],
                max_score_vector=self.max_score_vector,
                validate=False,
            )
            m.calibrate()
        return m

    @staticmethod
    def _global_fit_params(m, small_sample=False):
        """(labels, estimates, covariance, comparable mask, location blocks) for
        global_fit(): all uncentred thresholds; those next to a category
        unused in this group are marked not comparable."""
        x = m.responses.to_numpy(dtype=float)
        labels, est, ok = [], [], []
        for i, item in enumerate(m.item_names):
            thr = m.thresholds_uncentred.loc[item].dropna().values.astype(float)
            for k in range(1, int(m.max_score_vector[item]) + 1):
                labels.append(f"{item}, threshold {k}")
                est.append(thr[k - 1])
                ok.append((x[:, i] == k).sum() > 0 and (x[:, i] == k - 1).sum() > 0)
        est = np.array(est)
        V, _ = m._sandwich_covariance(small_sample=small_sample)
        return labels, est - est.mean(), V, np.array(ok), [np.arange(len(est))]

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
        is disabled (raises NotImplementedError).
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
        warnings.warn(
            "andersen_lr_test() is deprecated; use global_fit() (Wald), which is "
            "calibrated for any split, including split='exogenous' with "
            "covariate=. global_fit(method='andersen-lr') is reserved for "
            "conditional maximum likelihood estimates (planned).",
            FutureWarning,
            stacklevel=2,
        )
        from raschpy.pcm import PCM

        if split_by not in ("person_location", "score", "exogenous"):
            raise ValueError(
                "split_by must be 'person_location', 'score', or 'exogenous'"
            )

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
            self.calibrate(
                constant=constant, method=method, robust=robust, log_lik_tol=log_lik_tol
            )
        if not hasattr(self, "persons"):
            self.person_estimates(
                warm_corr=warm_corr,
                tolerance=tolerance,
                max_iters=max_iters,
                ext_score_adjustment=ext_score_adjustment,
            )

        scores = self.responses.sum(axis=1)
        max_scores = (
            self.responses.notna().mul(self.max_score_vector, axis=1).sum(axis=1)
        )
        non_extreme = self.responses.index[(scores > 0) & (scores < max_scores)]

        group_idx = self._resolve_andersen_groups(
            split_by, covariate, non_extreme, self.persons, scores
        )

        # Full-model LL restricted to persons in either group
        combined_idx = group_idx[list(group_idx)[0]].append(
            group_idx[list(group_idx)[1]]
        )

        # Re-estimate full model on combined subset so the LR comparison is fair
        m_full = PCM(self.responses.loc[combined_idx], self.max_score_vector)
        m_full.calibrate(
            constant=constant, method=method, robust=robust, log_lik_tol=log_lik_tol
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
            m = PCM(self.responses.loc[idx], self.max_score_vector)
            m.calibrate(
                constant=constant, method=method, robust=robust, log_lik_tol=log_lik_tol
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
        dcf_cap=15000,
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
            test='lr'/'both' is mildly liberal; test='wald' (the default)
            is preferred.
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
            to guard against the opposite problem from welch's: at very
            large N, SE shrinks enough that trivial differences become
            "significant", swamping the logit-magnitude thresholds'
            intent. The two components use different mechanisms, because
            they were found to need them (see reference_n/dcf_cap below):
            item-location SE is rescaled by sqrt(actual_n / reference_n)
            unconditionally (Tristán, 2006, Rasch Measurement Transactions
            20:3); threshold-cell SE is rescaled by a per-cell dynamic cap
            on its own harmonic-mean information weight (dcf_cap), applied
            only above the cap. Each group's own SE is rescaled
            independently, then combined as usual. Applied to the
            Wald/Welch test, the category tests, and (if welch=True) the
            Welch-Satterthwaite df, which then also uses the adjusted n
            rather than each group's actual n, for internal consistency.
        reference_n : int, default 100
            Reference sample size for size_adjust=True's item-location
            component only (dif_table) — does not affect threshold_dif_table,
            which uses dcf_cap instead (see below). 100 is Tristán's own
            default recommendation; ~60 is suggested there for closer
            alignment with the ETS Category B boundary specifically
            (100*(0.43/0.55)**2). Unused unless size_adjust=True. Checked
            directly by simulation (500-6000/group, 0.8-logit item-location
            shift): no practical power blind spot in that range, unlike the
            fixed-reference_n approach this package used for threshold-cell
            DCF before dcf_cap replaced it (see dcf_cap).
        dcf_cap : float, default 15000
            Cap, in harmonic-mean information-weight units (NOT a person
            count, unlike reference_n), for size_adjust=True's
            threshold-cell (DCF) component only — does not affect
            dif_table. Per item-threshold cell, independently for each
            group: effective_weight = min(harmonic_weight, dcf_cap);
            se_adjusted = se_actual * max(1, sqrt(harmonic_weight/dcf_cap)),
            where harmonic_weight is that cell's own row-summed harmonic
            mean of the raw pairwise comparison matrix and its transpose
            (see _pairwise_harmonic_weights — the same Bradley-Terry
            Fisher-information-weighted idea behind RSM's CPAT, expressed
            in PCM's cross-item comparison structure). Below the cap, no
            adjustment at all — unlike reference_n's unconditional rescale,
            an ordinary-sized study's own SE is used as-is. This replaces
            an earlier design (a single fixed reference_n shared with
            dif_table) that was found by direct simulation to make DCF
            power collapse to ~0% by N=1500/group at real effect sizes,
            because threshold-cell information doesn't scale with raw
            person count the way item-location information does, and no
            single fixed reference_n could serve both. Calibrated by
            simulation across max_score in {3, 5, 8} (I=15, 0.8-logit
            category-width shift, N in {500,...,6000}/group): power
            saturates at cap>=~2000 and Type-I stays at/below nominal
            throughout that whole range, consistently across all three
            max_score values tested — see the raschpy_dynamic_n_calibration
            project notes for the full simulation record. Not yet
            recalibrated for designs far outside that swept range (e.g.
            much larger max_score, or badly mixed max_score_vector
            designs) — 15000 is a reasonable, validated default, not a
            provably-universal one.
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

        self._dif_validate(
            correction,
            selection_method,
            test,
            category_thresholds,
            reference_n,
            dcf_cap=dcf_cap,
        )

        # Non-extreme persons, matching andersen_lr_test's convention
        scores = self.responses.sum(axis=1)
        max_scores = (
            self.responses.notna().mul(self.max_score_vector, axis=1).sum(axis=1)
        )
        non_extreme = self.responses.index[(scores > 0) & (scores < max_scores)]

        cov_values, reference, focal_levels, group_sizes, n_missing = (
            self._dif_group_setup(non_extreme, covariate, reference)
        )

        needs_ll = test in ("lr", "both") or omnibus
        pe_kw = dict(
            warm_corr=warm_corr,
            tolerance=tolerance,
            max_iters=max_iters,
            ext_score_adjustment=ext_score_adjustment,
        )

        ref_idx = cov_values.index[cov_values == reference]
        ref_model = PCM(self.responses.loc[ref_idx], self.max_score_vector)
        ref_model.calibrate(
            constant=constant, method=method, robust=robust, log_lik_tol=log_lik_tol
        )
        ref_model.std_errors(
            no_of_samples=no_of_samples,
            constant=constant,
            method=method,
            log_lik_tol=log_lik_tol,
            seed=seed,
        )
        if needs_ll:
            ref_model.person_estimates(**pe_kw)
            ll_ref = ref_model._log_likelihood()
        ref_hweights = ref_model._pairwise_harmonic_weights() if size_adjust else None

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
            focal_model.calibrate(
                constant=constant, method=method, robust=robust, log_lik_tol=log_lik_tol
            )
            focal_model.std_errors(
                no_of_samples=no_of_samples,
                constant=constant,
                method=method,
                log_lik_tol=log_lik_tol,
                seed=seed,
            )
            if needs_ll:
                focal_model.person_estimates(**pe_kw)
                ll_focal = focal_model._log_likelihood()

            # --- Item-location DIF ---
            wald_result = self._dif_item_wald_block(
                ref_model,
                focal_model,
                ref_model.central_bootstrap,
                focal_model.central_bootstrap,
                selection_method,
                corr_tol,
                sd_ratio_tol,
                min_anchors,
                wald_alpha,
                size_adjust,
                reference_n,
                welch,
                correction,
                alpha,
                logit_threshold,
                category,
                category_thresholds,
                category_alpha,
                ref_cov=(
                    ref_model._central_covariance()
                    if ref_model.se_method == "sandwich"
                    else None
                ),
                focal_cov=(
                    focal_model._central_covariance()
                    if focal_model.se_method == "sandwich"
                    else None
                ),
            )
            common_items = wald_result["common_items"]
            tc = wald_result["tc"]
            focal_shifted = wald_result["focal_shifted"]
            diff = wald_result["diff"]
            se = wald_result["se"]
            z = wald_result["z"]
            p = wald_result["p"]
            p_corrected = wald_result["p_corrected"]
            flagged = wald_result["flagged"]
            selected = wald_result["selected"]
            item_welch_df = wald_result["welch_df"]
            item_category_col = wald_result["category_col"]
            anchor_selection_dict[focal] = wald_result["anchor_table"]

            if omnibus:
                combined_idx = ref_idx.append(focal_idx)
                m_full = PCM(self.responses.loc[combined_idx], self.max_score_vector)
                m_full.calibrate(
                    constant=constant,
                    method=method,
                    robust=robust,
                    log_lik_tol=log_lik_tol,
                )
                m_full.person_estimates(**pe_kw)
                ll_full = m_full._log_likelihood()
                pooled_ref_persons = m_full.persons.reindex(ref_model.responses.index)
                pooled_focal_persons = m_full.persons.reindex(
                    focal_model.responses.index
                )
                if omnibus_scope == "item":
                    pooled_step_shape = m_full.thresholds_uncentred.sub(
                        m_full.items, axis=0
                    )
                    ll_ref_omni = self._item_location_log_likelihood(
                        ref_model.responses,
                        pooled_ref_persons,
                        pooled_step_shape,
                        list(common_items),
                        tolerance=tolerance,
                        max_iters=max_iters,
                    )
                    ll_focal_omni = self._item_location_log_likelihood(
                        focal_model.responses,
                        pooled_focal_persons,
                        pooled_step_shape,
                        list(common_items),
                        tolerance=tolerance,
                        max_iters=max_iters,
                    )
                    df_omni = len(common_items) - 1
                else:
                    ll_ref_omni = ref_model._log_likelihood(persons=pooled_ref_persons)
                    ll_focal_omni = focal_model._log_likelihood(
                        persons=pooled_focal_persons
                    )
                    df_omni = (
                        len(common_items)
                        - 1
                        + sum(
                            int(self.max_score_vector[item]) - 1
                            for item in common_items
                        )
                    )
                lr_omni = max(0.0, -2 * (ll_full - (ll_ref_omni + ll_focal_omni)))
                p_omni = float(chi2.sf(lr_omni, df_omni))
                omnibus_rows[focal] = {
                    "LR": lr_omni,
                    "df": df_omni,
                    "p": p_omni,
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
                    pooled_focal_scale = pooled - tc[item]

                    ref_scratch.thresholds_uncentred = (
                        ref_model.thresholds_uncentred.copy()
                    )
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
                    row_f[: len(focal_offsets)] = (
                        pooled_focal_scale + focal_offsets.values
                    )
                    focal_scratch.thresholds_uncentred.loc[item] = row_f
                    focal_scratch.person_estimates(**pe_kw)
                    ll_focal_h0 = focal_scratch._log_likelihood()

                    lr_i = max(0.0, 2 * (ll_h1 - (ll_ref_h0 + ll_focal_h0)))
                    lr_rows[item] = {"LR": lr_i, "p_LR": float(chi2.sf(lr_i, 1))}

                lr_table = self._dif_lr_table(
                    lr_rows, common_items, correction, alpha, diff, logit_threshold
                )

            item_table = self._dif_item_table(
                focal,
                ref_model,
                focal_model,
                common_items,
                focal_shifted,
                diff,
                se,
                z,
                p,
                p_corrected,
                selected,
                flagged,
                welch_df=item_welch_df if welch else None,
                category_col=item_category_col if category else None,
                lr_table=lr_table if test in ("lr", "both") else None,
            )

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
                    # DCF dynamic cap: per-cell harmonic-mean information
                    # weight (not raw person count) is the effective n here,
                    # capped rather than pinned -- see dcf_cap docstring for
                    # why a fixed reference_n (as used for item-location DIF
                    # above) was found to make DCF power collapse at real
                    # sample sizes instead.
                    focal_hweights = focal_model._pairwise_harmonic_weights()
                    row_index = pd.MultiIndex.from_frame(
                        threshold_table[["Item", "Category"]]
                    )
                    ref_hw = ref_hweights.reindex(row_index).values
                    focal_hw = focal_hweights.reindex(row_index).values
                    ref_eff = np.minimum(ref_hw, dcf_cap)
                    focal_eff = np.minimum(focal_hw, dcf_cap)
                    threshold_table["_ref_se"] = threshold_table[
                        "_ref_se"
                    ].values * np.sqrt(ref_hw / ref_eff)
                    threshold_table["_focal_se"] = threshold_table[
                        "_focal_se"
                    ].values * np.sqrt(focal_hw / focal_eff)
                    threshold_table["SE"] = np.sqrt(
                        threshold_table["_ref_se"] ** 2
                        + threshold_table["_focal_se"] ** 2
                    )
                    ref_n_thr = ref_eff
                    focal_n_thr = focal_eff

                if welch:
                    thr_z_vals, thr_df_vals, thr_p_vals = self._welch_satterthwaite(
                        threshold_table["Difference"].values,
                        threshold_table["_ref_se"].values,
                        ref_n_thr,
                        threshold_table["_focal_se"].values,
                        focal_n_thr,
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
        self.dif_omnibus_table = pd.DataFrame(omnibus_rows).T if omnibus else None
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
            min_effect is optional and a matter of taste: in simulation the
            tests alone classify well, so the gate expresses a judgement of
            practical importance rather than improving accuracy. If wanted,
            0.43 and 0.64 mirror the ETS DIF category boundaries (moderate
            and large).
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
        if (
            sampling is not None
            and sampling != "dynamic"
            and not isinstance(sampling, int)
        ):
            raise ValueError("sampling must be None, 'dynamic', or an integer")

        if not (self.max_score_vector == self.max_score_vector.iloc[0]).all():
            raise ValueError(
                "RSM vs PCM comparison requires uniform max scores across items."
            )
        max_score = int(self.max_score_vector.iloc[0])

        if not hasattr(self, "thresholds_uncentred"):
            self.calibrate(
                constant=constant, method=method, robust=robust, log_lik_tol=log_lik_tol
            )
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

        effect = float(
            self.thresholds.sub(rsm.thresholds, axis=1).abs().to_numpy().max()
        )
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
                {
                    "LR statistic": lr,
                    "df": df,
                    "p-value": p,
                    "Effect": effect,
                    "Preferred": preferred,
                },
                name="RSM vs PCM LR test",
            )

        elif test == "AIC":
            aic_pcm = 2 * k_pcm - 2 * ll_pcm
            aic_rsm = 2 * k_rsm - 2 * ll_rsm
            self.model_comparison_rsm_pcm_aic = {"PCM": aic_pcm, "RSM": aic_rsm}

            if aic_sig_test:
                delta = aic_rsm - aic_pcm
                aic_p = float(np.exp(-abs(delta) / 2))
                preferred = (
                    "PCM"
                    if (delta > 0 and aic_p < alpha and effect >= min_effect)
                    else "RSM"
                )
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
                preferred = (
                    "PCM" if (aic_pcm < aic_rsm and effect >= min_effect) else "RSM"
                )
                self.model_comparison_rsm_pcm_aic_preferred = preferred
                self.model_comparison_rsm_pcm_aic_summary = pd.Series(
                    {
                        "PCM AIC": aic_pcm,
                        "RSM AIC": aic_rsm,
                        "Effect": effect,
                        "Preferred": preferred,
                    },
                    name="RSM vs PCM AIC comparison",
                )

        elif test == "BIC":
            bic_pcm = k_pcm * np.log(n_ll) - 2 * ll_pcm
            bic_rsm = k_rsm * np.log(n_ll) - 2 * ll_rsm
            preferred = "PCM" if (bic_pcm < bic_rsm and effect >= min_effect) else "RSM"
            self.model_comparison_rsm_pcm_bic = {"PCM": bic_pcm, "RSM": bic_rsm}
            self.model_comparison_rsm_pcm_bic_preferred = preferred
            self.model_comparison_rsm_pcm_bic_summary = pd.Series(
                {
                    "PCM BIC": bic_pcm,
                    "RSM BIC": bic_rsm,
                    "Effect": effect,
                    "Preferred": preferred,
                },
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
        filename=None,
        file_format="csv",
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
        filename : str or None, default None
            If given, also saves the returned table to
            ``{filename}.{file_format}`` (path without extension, as for
            the plotting methods).
        file_format : {'csv', 'xlsx'}, default 'csv'
            File format used when filename is given.

        Attributes set
        --------------
        item_stats : pandas.DataFrame
            Item statistics table with items as rows. Always contains
            'Estimate', 'Count', 'Facility', 'Infit MS', 'Outfit MS'.
            Optional: 'SE' and CI bounds (if se=True), 'Infit Z',
            'Outfit Z', 'PM corr', 'Exp PM corr'.

        Returns
        -------
        pandas.DataFrame
            The table, also stored as the attribute listed above.
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
        _save_table(self.item_stats, filename, file_format)
        return self.item_stats

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
        filename=None,
        file_format="csv",
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
        filename : str or None, default None
            If given, also saves the returned table to
            ``{filename}.{file_format}`` (path without extension, as for
            the plotting methods).
        file_format : {'csv', 'xlsx'}, default 'csv'
            File format used when filename is given.

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
        pandas.DataFrame
            The table, also stored as the attribute listed above.
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
        _save_table(self.threshold_stats, filename, file_format)
        return self.threshold_stats

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
        filename=None,
        file_format="csv",
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
        filename : str or None, default None
            If given, also saves the returned table to
            ``{filename}.{file_format}`` (path without extension, as for
            the plotting methods).
        file_format : {'csv', 'xlsx'}, default 'csv'
            File format used when filename is given.

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

        Returns
        -------
        pandas.DataFrame
            The table, also stored as the attribute listed above.
        """
        if not hasattr(self, "thresholds"):
            self.calibrate(
                constant=constant, method=method, robust=robust, log_lik_tol=log_lik_tol
            )
        if not hasattr(self, "cat_width_se"):
            self.std_errors(
                interval=interval,
                no_of_samples=no_of_samples,
                constant=constant,
                method=method,
                matrix_power=matrix_power,
                log_lik_tol=log_lik_tol,
                seed=seed,
            )

        cat_items = [
            item
            for item in self.responses.columns
            if len(self.thresholds.loc[item].dropna()) >= 2
        ]
        if cat_items:
            cat_widths = {
                item: self.thresholds.loc[item].dropna().diff().dropna().values
                for item in cat_items
            }
            cat_mi = pd.MultiIndex.from_tuples(
                [
                    (item, k)
                    for item in cat_items
                    for k in range(1, len(cat_widths[item]) + 1)
                ]
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
            stats["Disordered"] = (
                np.concatenate([cat_widths[item] for item in cat_items]) < 0
            )
            if getattr(self, "cat_width_bootstrap", None) is not None:
                prop = np.concatenate(
                    [
                        (self.cat_width_bootstrap[item] < 0).mean(axis=0).values
                        for item in cat_items
                    ]
                )
            else:  # sandwich SEs: normal-approximation probability of a negative width
                prop = np.concatenate(
                    [
                        norm.cdf(
                            -np.asarray(cat_widths[item], float)
                            / np.asarray(self.cat_width_se[item], float)
                        )
                        for item in cat_items
                    ]
                )
            stats["Prop disordered"] = prop.round(dp)
        else:
            stats = pd.DataFrame(
                columns=["Estimate", "SE", "Disordered", "Prop disordered"]
            )
        self.category_stats = stats
        _save_table(self.category_stats, filename, file_format)
        return self.category_stats

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
        filename=None,
        file_format="csv",
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
        filename : str or None, default None
            If given, also saves the returned table to
            ``{filename}.{file_format}`` (path without extension, as for
            the plotting methods).
        file_format : {'csv', 'xlsx'}, default 'csv'
            File format used when filename is given.

        Attributes set
        --------------
        person_stats : pandas.DataFrame
            Person statistics with persons as rows. Contains 'Estimate',
            'CSEM', 'Score', 'Max score', 'p', 'Infit MS', 'Infit Z',
            'Outfit MS', 'Outfit Z'. Optional: 'RSEM'.

        Returns
        -------
        pandas.DataFrame
            The table, also stored as the attribute listed above.
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
        _save_table(self.person_stats, filename, file_format)
        return self.person_stats

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
        filename=None,
        file_format="csv",
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
        filename : str or None, default None
            If given, also saves the returned table to
            ``{filename}.{file_format}`` (path without extension, as for
            the plotting methods).
        file_format : {'csv', 'xlsx'}, default 'csv'
            File format used when filename is given.

        Attributes set
        --------------
        test_stats : pandas.DataFrame
            Three-column table (Items, Thresholds, Persons) with rows:
            Mean, SD, Separation ratio, Strata, Reliability.

        Returns
        -------
        pandas.DataFrame
            The table, also stored as the attribute listed above.
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

        items_col = [
            self.items.mean(),
            self.items.std(),
            self.isi_central,
            self.item_strata,
            self.item_reliability,
        ]
        thr_col = [
            self.threshold_list.mean(),
            self.threshold_list.std(),
            self.isi_thresholds,
            self.threshold_strata,
            self.threshold_reliability,
        ]
        persons_col = [
            self.persons.mean(),
            self.persons.std(),
            self.psi,
            self.person_strata,
            self.person_reliability,
        ]
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
        _save_table(self.test_stats, filename, file_format)
        return self.test_stats

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
                        (person_locations_subset >= q.values[i])
                        & (person_locations_subset < q.values[i + 1])
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
                "#E69F00",
                "#56B4E9",
                "#009E73",
                "#F0E442",
                "#0072B2",
                "#D55E00",
                "#CC79A7",
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
                [
                    path_effects.withStroke(
                        linewidth=2.5,
                        foreground=(
                            "white"
                            if plot_style in self._DARK_BACKGROUND_STYLES
                            else "black"
                        ),
                    )
                ]
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
                    x_data,
                    y_data[:, i],
                    "",
                    color=col,
                    label=label,
                    path_effects=line_fx,
                )
                if max_score_lines is not None:
                    ax.axhline(
                        max_score_lines[i], color=col, linestyle=":", linewidth=1
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
                        x_obs_data,
                        y_obs_data,
                        color=col,
                        s=40,
                        alpha=0.7,
                        edgecolors="k" if marker_border else "none",
                        zorder=3,
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
                                xd,
                                y_obs_data[:, j],
                                color=col,
                                s=40,
                                alpha=0.7,
                                edgecolors="k" if marker_border else "none",
                                zorder=3,
                            )
                    except Exception:
                        pass

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
                        f"{it}: {round(xval, 2)}"
                        if items_multi
                        else str(round(xval, 2))
                    )
                    if central_location_fit and items_multi:
                        label_y = (
                            curve_peak + gap + label_h_data * (n_labels - idx - 0.5)
                        )
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
                                    x_min + (x_max - x_min) / 100,
                                    s + y_max / 50,
                                    str(s),
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
                            x=estimate,
                            ymin=-100,
                            ymax=info,
                            color=colorVal,
                            linestyles="dashed",
                        )
                        ax.hlines(
                            y=info,
                            xmin=-100,
                            xmax=estimate,
                            color=colorVal,
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
                                    label_x,
                                    info + y_max / 50,
                                    f"{it}: {round(info, 3)}",
                                    color=colorVal,
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
                        items=[it],
                        person_locations=self.persons,
                        no_of_classes=no_of_classes,
                    )
                    if xobsdata is None:
                        xobsdata = pd.Series(mpl)
                    yobs_cols.append(np.array(om))
                yobsdata = np.column_stack(yobs_cols)
            else:
                mean_person_locations, obs_means = self.class_intervals(
                    items=item,
                    person_locations=self.persons,
                    no_of_classes=no_of_classes,
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
                    self.variance_uncentred(
                        a, self.thresholds_uncentred.loc[item].dropna()
                    )
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
        dpi=300,
        plot_density=None,
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
        if plot_density is not None:
            warnings.warn(
                "plot_density= is deprecated and will be removed in a future version; use dpi= instead.",
                FutureWarning,
                stacklevel=2,
            )
            dpi = plot_density
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
            plot_density=dpi,
        )

    # ------------------------------------------------------------------
    # Simulation recovery diagnostics
    # ------------------------------------------------------------------

    @staticmethod
    def _item_category_widths(thresholds_uncentred):
        """
        Per-item category widths from an items x threshold-position
        DataFrame (NaN-padded per item's own max_score_vector[item]),
        returned in the same items x width-position shape (width position
        k = threshold position k+1 minus threshold position k, relabelled
        1..len-1 per item -- matches cat_width_bootstrap's own convention
        in std_errors()).
        """
        widths = {}
        for item in thresholds_uncentred.index:
            row = thresholds_uncentred.loc[item].dropna()
            w = row.diff().dropna()
            w.index = range(1, len(row))
            widths[item] = w
        return pd.DataFrame(widths).T

    def _flatten_per_item(self, template, values_by_item):
        """
        Build a flat (threshold_position, item) Series matching
        template.unstack().dropna()'s index exactly, from a dict of
        {item: array-like of per-item values in template's own column
        order for that item}. Used to align self.threshold_se/
        self.cat_width_se (plain per-item arrays, no own index) with
        self.thresholds_uncentred/category-widths' own flattened index.
        """
        filled = pd.DataFrame(
            index=template.index, columns=template.columns, dtype=float
        )
        for item in template.index:
            valid_cols = template.loc[item].dropna().index
            vals = np.asarray(values_by_item[item], dtype=float).ravel()
            filled.loc[item, valid_cols] = vals[: len(valid_cols)]
        return filled.unstack().dropna()

    def recovery_stats(
        self,
        anchor=False,
        alpha=0.05,
        correction="bh",
        divergence_tol=0.1,
        plot=True,
        **plot_kwargs,
    ):
        """
        Compare recovered parameter estimates against the true generating
        values from the simulation this object was constructed from.

        No-op (warns, returns None) unless this object was built from a
        simulation (self.generating exists) -- NOT restricted to
        self.generating.model == 'PCM'. Person recovery is valid across
        any appropriate model pairing (e.g. this PCM's persons compared
        against pseudo-estimates built from RSM_Sim-generated data), so
        it's always attempted. Item recovery is always attempted too
        (item difficulties are comparable across SLM/PCM/RSM regardless
        of threshold structure).

        The PCM<->RSM pairing specifically is stricter than a per-
        parameter-type skip: if the generating side is an RSM_Sim,
        NOTHING fires at all (the whole method no-ops, not just
        thresholds) unless EVERY max score matches -- this PCM object's
        own max_score_vector must be constant across items AND equal
        RSM_Sim's own max_score. A partial mismatch means the two models
        aren't testing the same category structure at all, not merely a
        threshold-shape mismatch, so allowing items/persons through while
        just skipping thresholds would be misleading. When every max
        score does match, thresholds/category_widths compare this PCM's
        own per-item thresholds against RSM's single shared vector,
        broadcast across every item. Against SLM-generated data (no
        thresholds at all), thresholds/category_widths are skipped with a
        warning (self.recovery.thresholds/.category_widths left as None,
        items/persons still proceed) rather than raising or fabricating a
        meaningless comparison -- unlike the RSM case above, SLM doesn't
        trigger a whole-method no-op, since SLM's items are still
        directly comparable regardless of its lack of thresholds.

        Person recovery does NOT compare self.persons directly against
        self.generating.persons -- that would conflate threshold
        estimation and model-specification error with the irreducible
        stochastic noise of a single finite response pattern (even a
        perfect model, perfectly calibrated, shows nonzero residuals
        against true theta, purely from response-pattern discreteness).
        Instead, following Elliott & Buttery (2022a), a pseudo-estimate is
        computed: the same Newton-Raphson person() procedure re-run on the
        same self.responses, but using the TRUE generating (uncentred)
        thresholds in place of the estimated ones. The persons table's
        'error' column is (recovered estimate - pseudo-estimate), not
        (recovered estimate - true theta) -- both share the same actual
        response-pattern noise, so differencing cancels that shared
        stochastic component. What remains is the COMBINED effect of (a)
        threshold parameter estimation error and (b) any model
        misspecification (fitting a model form that doesn't match how the
        data were actually generated) on person measurement -- this
        method does not, and cannot, separate those two; Elliott & Buttery
        (2022a) used exactly this construction to demonstrate the
        person-measurement impact of (b), not just (a). Its 'generating'
        column holds this pseudo-estimate, not the raw true theta --
        unlike every other parameter type's 'generating' column, which
        does hold the actual true value.

        Parameters
        ----------
        anchor : bool, default False
            If True, compare self.anchor_items (from calibrate_anchor())
            against self.generating.items, instead of self.items. Raises
            ValueError if calibrate_anchor() has not been run. PCM has no
            anchored threshold/person-estimate attributes (only the
            diagnostic anchor_threshold_test(_aggregate)), so thresholds/
            category_widths/persons recovery are unaffected by this flag
            either way.
        alpha : float, default 0.05
            Significance threshold for the per-parameter z-test
            (Flagged/Flagged_corrected).
        correction : {'bh', 'bonferroni', None}, default 'bh'
            Multiple-comparison correction, applied within each parameter
            type's own family of tests (matches dif_test's convention).
        divergence_tol : float, default 0.1
            Threshold for the OLS-vs-Huber slope divergence flag.
        plot : bool, default True
            If True, also produce a recovery plot per parameter type via
            self._recovery_plot().
        **plot_kwargs
            Forwarded to _recovery_plot() (filename, file_format, dpi,
            plot_style, font, title_font_size, axis_font_size, labelsize,
            figsize).

        Returns
        -------
        None
            Results are stored in self.recovery, not returned directly --
            a namespace with .items/.persons/.thresholds/.category_widths,
            each holding .table (a per-parameter DataFrame: generating,
            estimated, SE, error, abs_error, z, p, p_corrected, Flagged,
            Flagged_corrected) and .summary (a DataFrame, column
            'Overall': RMSE, MAE, bias, r, r_squared, SD_ratio, OLS_slope,
            OLS_intercept, Huber_slope, Huber_intercept, Divergence_flag,
            Prop_flagged). thresholds/category_widths tables are indexed
            by (threshold_position, item) -- self.thresholds_uncentred.
            unstack().dropna()'s own convention. Also sets
            self.recovery.parameter_estimation_residual (float, or None
            when thresholds/category_widths are skipped): the Elliott &
            Buttery (2022) expected-score-space RMS residual between
            generating and estimated (uncentred) thresholds at the true
            generating theta -- see _parameter_estimation_residual.
        """
        from raschpy.base import _SimParams

        if not hasattr(self, "generating"):
            warnings.warn(
                "recovery_stats() is a no-op: this object was not built "
                "from a simulation (no self.generating found).",
                UserWarning,
                stacklevel=2,
            )
            return None

        if anchor and not hasattr(self, "anchor_items"):
            raise ValueError(
                "anchor=True requires calibrate_anchor() to have been run "
                "first (no self.anchor_items found)."
            )

        gen_model = self.generating.model
        if gen_model == "RSM":
            # Cross-model PCM<->RSM comparison: nothing fires at all
            # (not just thresholds) unless EVERY max score matches -- this
            # PCM object's own max_score_vector must be constant AND
            # equal RSM_Sim's own max_score. A partial mismatch means the
            # two models aren't testing the same category structure at
            # all, not just a threshold-shape mismatch.
            own_msv = self.max_score_vector
            own_values = (
                set(own_msv.values) if hasattr(own_msv, "values") else set(own_msv)
            )
            if len(own_values) != 1 or list(own_values)[0] != self.generating.max_score:
                warnings.warn(
                    "recovery_stats() is a no-op: PCM<->RSM comparison "
                    "requires every max score to match -- this PCM "
                    f"object's own max_score_vector={dict(own_msv) if hasattr(own_msv, 'items') else list(own_msv)} "
                    "must be constant and equal RSM_Sim's max_score="
                    f"{self.generating.max_score}.",
                    UserWarning,
                    stacklevel=2,
                )
                return None
        elif gen_model == "SLM":
            # SLM_Sim's data is genuinely dichotomous -- only a valid
            # nested comparison when this PCM object's own
            # max_score_vector is also constant at 1 (same "every max
            # score must match" rule as the RSM case above, SLM
            # implicitly being max_score=1).
            own_msv = self.max_score_vector
            own_values = (
                set(own_msv.values) if hasattr(own_msv, "values") else set(own_msv)
            )
            if own_values != {1}:
                warnings.warn(
                    "recovery_stats() is a no-op: SLM_Sim-generated data "
                    "is dichotomous, only a valid nested comparison when "
                    "this PCM's own max_score_vector is constant at 1 "
                    f"(got {dict(own_msv) if hasattr(own_msv, 'items') else list(own_msv)}).",
                    UserWarning,
                    stacklevel=2,
                )
                return None

        # interval=0.95 here (not strictly needed by recovery_stats() itself,
        # which only reads item_se/csem_vector) so a later full=True call to
        # item_stats_df()/etc. doesn't crash: those check hasattr(self,
        # "item_se") to decide whether to (re-)bootstrap, and would otherwise
        # find item_se already set but item_low/item_high still missing --
        # a real crash discovered via testing, not a hypothetical.
        if not hasattr(self, "item_se"):
            self.std_errors(interval=0.95)
        if not hasattr(self, "csem_vector"):
            self.fit_statistics(interval=0.95)

        self.recovery = _SimParams()

        est_items = self.anchor_items if anchor else self.items
        items_table, items_summary = self._parameter_recovery_table(
            self.generating.items,
            est_items,
            se=self.item_se,
            alpha=alpha,
            correction=correction,
            divergence_tol=divergence_tol,
        )
        self.recovery.items = _SimParams()
        self.recovery.items.table = items_table
        self.recovery.items.summary = items_summary

        gen_uncentred = None
        if gen_model == "PCM":
            # PCM_Sim's own thresholds_uncentred can use a different
            # column-label convention (e.g. 0-indexed) than the analysis
            # class's own (1..max_score) -- relabel onto the analysis
            # class's own columns before unstacking, so the two sides
            # align by threshold POSITION within each item's row rather
            # than by a raw label that may not mean the same thing on both
            # sides. Confirmed empirically this matters: without it, a
            # plain .unstack().dropna() union produces spurious extra rows
            # from labels that only exist on one side.
            gen_uncentred = self.generating.thresholds_uncentred.copy()
            gen_uncentred.columns = self.thresholds_uncentred.columns
        elif gen_model == "RSM":
            # Max-score match already verified above -- safe to proceed.
            # RSM's single shared vector broadcasts across every item
            # this PCM object has.
            gen_uncentred = pd.DataFrame(
                {item: self.generating.thresholds for item in self.item_names}
            ).T
            gen_uncentred.columns = self.thresholds_uncentred.columns
        else:
            warnings.warn(
                f"recovery_stats(): thresholds/category_widths skipped "
                f"-- generating model {gen_model!r} has no threshold "
                f"structure comparable to PCM's.",
                UserWarning,
                stacklevel=2,
            )

        self.recovery.thresholds = None
        self.recovery.category_widths = None
        gen_thresh = est_thresh = gen_widths = est_widths = None
        if gen_uncentred is not None:
            gen_thresh = gen_uncentred.unstack().dropna()
            est_thresh = self.thresholds_uncentred.unstack().dropna()
            threshold_se = self._flatten_per_item(
                self.thresholds_uncentred, self.threshold_se
            )
            thresholds_table, thresholds_summary = self._parameter_recovery_table(
                gen_thresh,
                est_thresh,
                se=threshold_se,
                alpha=alpha,
                correction=correction,
                divergence_tol=divergence_tol,
            )
            self.recovery.thresholds = _SimParams()
            self.recovery.thresholds.table = thresholds_table
            self.recovery.thresholds.summary = thresholds_summary

            gen_widths_wide = self._item_category_widths(gen_uncentred)
            est_widths_wide = self._item_category_widths(self.thresholds_uncentred)
            gen_widths = gen_widths_wide.unstack().dropna()
            est_widths = est_widths_wide.unstack().dropna()
            cat_width_se = self._flatten_per_item(est_widths_wide, self.cat_width_se)
            widths_table, widths_summary = self._parameter_recovery_table(
                gen_widths,
                est_widths,
                se=cat_width_se,
                alpha=alpha,
                correction=correction,
                divergence_tol=divergence_tol,
            )
            self.recovery.category_widths = _SimParams()
            self.recovery.category_widths.table = widths_table
            self.recovery.category_widths.summary = widths_summary

        # Persons -- pseudo-estimate construction (Elliott & Buttery, 2022a).
        # Dispatches on self.generating.model, so this works even when the
        # generating structure came from a different model class (see
        # _pseudo_estimate_persons).
        pseudo_estimate = self._pseudo_estimate_persons()

        persons_table, persons_summary = self._parameter_recovery_table(
            pseudo_estimate,
            self.persons,
            se=self.csem_vector,
            alpha=alpha,
            correction=correction,
            divergence_tol=divergence_tol,
        )
        self.recovery.persons = _SimParams()
        self.recovery.persons.table = persons_table
        self.recovery.persons.summary = persons_summary

        # Parameter-estimation residual (Elliott & Buttery, 2022): RMS
        # expected-score residual between generating and estimated
        # (uncentred, per-item absolute-scale) thresholds -- these already
        # encode item location jointly with threshold shape, so no
        # separate item-location term is needed. Only meaningful when
        # thresholds/category_widths themselves were (gen_uncentred is not
        # None) -- see _parameter_estimation_residual.
        self.recovery.parameter_estimation_residual = None
        if gen_uncentred is not None:
            gen_thresh_by_item = gen_uncentred
            est_thresh_by_item = self.thresholds_uncentred
            self.recovery.parameter_estimation_residual = (
                self._parameter_estimation_residual(
                    self.generating.persons,
                    lambda theta, item: self.exp_score_uncentred(
                        theta, gen_thresh_by_item.loc[item].dropna().values
                    ),
                    lambda theta, item: self.exp_score_uncentred(
                        theta, est_thresh_by_item.loc[item].dropna().values
                    ),
                )
            )

        if plot:
            self._recovery_plot(
                self.generating.items, est_items, "items", **plot_kwargs
            )
            if gen_uncentred is not None:
                self._recovery_plot(gen_thresh, est_thresh, "thresholds", **plot_kwargs)
                self._recovery_plot(
                    gen_widths, est_widths, "category widths", **plot_kwargs
                )
            self._recovery_plot(pseudo_estimate, self.persons, "persons", **plot_kwargs)

        return None
