from math import log, log10
import warnings
from scipy.stats import chi2, gaussian_kde, norm, t as t_dist
from scipy.optimize import brentq

import numpy as np
import pandas as pd

from raschpy import _rcl

from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from matplotlib import colors as colors
from matplotlib import cm as cmx
from matplotlib.patches import Rectangle
import seaborn as sns
import matplotlib.patheffects as path_effects

from raschpy.base import Rasch, _BootstrapWarnings, _save_table


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

        self.model = "RSM"

        # Sim-aware instantiation: store sim attributes in self.generating namespace
        from raschpy.simulation.rsm_sim import RSM_Sim
        from raschpy.simulation.slm_sim import SLM_Sim
        from raschpy.simulation.pcm_sim import PCM_Sim
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
        elif isinstance(responses, (SLM_Sim, PCM_Sim)):
            # Cross-model construction -- lets recovery_stats() compare an
            # RSM fit against another model class's generating data (e.g.
            # for a person-recovery pseudo-estimate check, or an item/
            # threshold nesting check when the pairing is genuinely
            # comparable). RSM has no single max_score attribute on these
            # sim classes to infer from, so it must be given explicitly.
            sim = responses
            self.generating = _SimParams()
            for attr, value in vars(sim).items():
                setattr(self.generating, attr, value)
            if max_score is None:
                raise ValueError(
                    f"max_score must be given explicitly when constructing "
                    f"RSM from a {type(sim).__name__} (RSM has no single "
                    f"shared max_score to infer from one)."
                )
            self.max_score = int(max_score)
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

            # check_data_connectivity() itself warns about disconnection,
            # structural zeros and weak connectivity (one warning per problem).

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

    # Empty-category smoothing rules: lambda = a * w ** b, clipped to [lo, hi],
    # where w is the estimated mean category width; fallback lambda when no
    # width can be estimated (max_score=2).
    _EMPTY_CATEGORY_RULES = {
        "cpat-cml": (0.1767, -3.3579, 0.005, 3.0, 0.2),
        "cpat-info-weighted": (0.0483, -4.0196, 0.001, 3.0, 0.07),
    }
    # The collapsed scale's mean width is about twice the true category width
    # (each category next to the merged boundary absorbs half the empty one,
    # and the merged step is twice as steep as the others).
    _COLLAPSED_WIDTH_FACTOR = 0.5
    _THRESHOLD_METHODS = ("cpat-cml", "cpat-info-weighted")

    def _cpat_counts(self, threshold, responses=None):
        """
        CPAT comparison counts for the distance between thresholds k and k+1.

        Returns num, den, each (I, I) with zero diagonals: num[i, j] counts
        persons scoring k on both items i and j, den[i, j] persons scoring k-1
        on item i and k+1 on item j. Same-item cells are not comparisons and
        are excluded.
        """
        x = np.array(self.responses if responses is None else responses, dtype=np.float64)
        at_k = (x == threshold).astype(np.float64)
        at_km1 = (x == threshold - 1).astype(np.float64)
        at_kp1 = (x == threshold + 1).astype(np.float64)
        num = at_k.T @ at_k
        den = at_km1.T @ at_kp1
        np.fill_diagonal(num, 0.0)
        np.fill_diagonal(den, 0.0)
        return num, den

    def _threshold_distance(self, threshold, item_locations, constant=0.03,
                            threshold_method="cpat-cml"):
        """
        Estimate the distance between adjacent Rasch-Andrich thresholds (CPAT).

        Implements Elliott & Buttery (2022). For threshold k, counts:
          num[i,j]: persons scoring exactly k on both items i and j
          den[i,j]: persons scoring k-1 on item i and k+1 on item j
        Conditioning on these patterns removes person location, leaving a
        contrast identifying the threshold distance. threshold_method selects
        the estimator: 'cpat-cml' (conditional maximum likelihood) or
        'cpat-info-weighted' (harmonic-mean information weighting, eq. 4.18).
        This is the plain estimator for one distance; threshold_set() adds
        the thin-data guard, empty-category smoothing and no-information
        handling.

        Parameters
        ----------
        threshold : int
            1-based threshold index (1 to max_score-1).
        item_locations : pandas.Series
            Item location estimates indexed by item name.
        constant : float, default 0.03
            Additive smoothing constant for observed item pairs.
        threshold_method : {'cpat-cml', 'cpat-info-weighted'}, default 'cpat-cml'

        Returns
        -------
        float
            Estimated distance between tau_threshold and tau_{threshold+1}, in logits.

        Raises
        ------
        ValueError
            If no item pair has any of the comparison patterns for this
            threshold (no information anywhere in the data to estimate it).
        """
        num, den = self._cpat_counts(threshold)
        est = item_locations.reindex(self.responses.columns).values.astype(float)
        loc = est[:, None] - est[None, :]
        if threshold_method == "cpat-info-weighted":
            distance = self._cpat_info_distance(num, den, loc, constant)
        else:
            distance = self._cpat_cml_distance(num, den, loc, constant)
        if not np.isfinite(distance):
            raise ValueError(self._no_info_message(threshold))
        return distance

    def _no_info_message(self, threshold):
        return (
            f"CPAT threshold distance for threshold {threshold} could not be "
            f"estimated: no pair of different items has any person scoring "
            f"{threshold} on both, or {threshold - 1} on one and "
            f"{threshold + 1} on the other, so the data contain no "
            f"information on the distance between thresholds {threshold} and "
            f"{threshold + 1}. The smoothing constant cannot supply it; a larger "
            f"or better-targeted sample is needed."
        )

    def _threshold_total_weights(self, constant=0.1):
        """
        CPAT's own total_weight (the harmonic-mean information weight
        _threshold_distance() sums over item pairs), for every threshold
        1..max_score-1 -- the shared-vector analogue of PCM's
        _pairwise_harmonic_weights(), used as the effective sample size for
        dif_test's DCF dynamic SE cap (dcf_cap). Mirrors the information-
        weighted estimator's own weight computation (not the distance itself),
        so this is cheap relative to a full calibrate() and doesn't need item
        locations.

        Parameters
        ----------
        constant : float, default 0.1
            Additive smoothing constant, matching threshold_constant from
            calibrate()/std_errors() for internal consistency.

        Returns
        -------
        pandas.Series
            total_weight per threshold, indexed 1..max_score-1 (matching
            cat_width_se's own indexing convention).
        """
        weights = {}
        for threshold in range(1, self.max_score):
            num_matrix, den_matrix = self._cpat_counts(threshold)
            valid = (num_matrix + den_matrix) > 0
            num_s = np.where(valid, num_matrix + constant, 0.0)
            den_s = np.where(valid, den_matrix + constant, 0.0)
            finite = valid & (num_s > 0) & (den_s > 0)
            weight_matrix = np.where(finite, 2.0 * num_s * den_s / (num_s + den_s), 0.0)
            weights[threshold] = weight_matrix.sum()
        return pd.Series(weights)

    def _collapsed_category_width(self, empty):
        """
        Mean category width from a quick fit with the empty middle categories
        merged into the category above (scores above each empty category
        shifted down by one). NaN if the collapsed scale has fewer than two
        thresholds.
        """
        new_max = self.max_score - len(empty)
        if new_max < 2:
            return np.nan
        collapsed = self.responses.copy()
        for category in sorted(empty, reverse=True):
            collapsed = collapsed.where(~(collapsed > category), collapsed - 1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            probe = RSM(collapsed, max_score=new_max, validate=False)
            probe.calibrate(
                estimation="pair-cpat",
                constant=getattr(self, "constant", 0.1),
                threshold_constant=0.03,
                threshold_method="cpat-cml",
            )
        widths = np.abs(np.diff(probe.thresholds.values.astype(float)))
        return float(np.mean(widths)) if len(widths) else np.nan

    def threshold_set(self, item_locations, constant=0.03, threshold_method="cpat-cml",
                      no_info="mean"):
        """
        Compute the Rasch-Andrich threshold vector from CPAT distances.

        Estimates each distance between adjacent thresholds, chains them via
        cumulative sum and mean-centres the result. Called by calibrate().

        With threshold_method='cpat-cml', a distance whose comparison
        patterns are observed fewer than CPAT_THIN_COUNT (20) times in total,
        summed over ordered pairs of different items, is estimated with the
        information-weighted estimator instead, which is more stable when
        the counts are that thin.

        If a middle score category has no responses, the three distances
        touching it are not identified by the data. They are estimated with
        a larger, targeted smoothing constant, lambda times the mean pair
        total, with lambda set by the estimated width of the other
        categories (narrower categories get more smoothing). This keeps the
        rest of the scale intact; only the point where the category
        response curves of the two categories either side of the empty one
        intersect is identified, not the two thresholds around it
        separately. A UserWarning describes the problem and the
        options.

        Parameters
        ----------
        item_locations : pandas.Series
            Item locations from Stage 1 (PAIR), indexed by item name.
        constant : float, default 0.03
            Additive smoothing constant for every adjacent-threshold link not
            touching an empty category.
        threshold_method : {'cpat-cml', 'cpat-info-weighted'}, default 'cpat-cml'
            'cpat-cml': conditional maximum likelihood (recommended).
            'cpat-info-weighted': the harmonic-mean information-weighted
            estimator of Elliott & Buttery (2022, eq. 4.18), which compresses
            the scale slightly in thin data.
        no_info : {'mean', 'raise'}, default 'mean'
            What to do if a distance has no information at all (no pair of
            different items with either comparison pattern). 'mean' sets it
            to the mean of the estimable distances (0 if none) and warns;
            'raise' raises a ValueError.

        Returns
        -------
        numpy.ndarray
            Threshold vector of length max_score, centred at 0.
        """
        if threshold_method not in self._THRESHOLD_METHODS:
            raise ValueError(
                f"threshold_method must be one of {self._THRESHOLD_METHODS}, "
                f"got {threshold_method!r}."
            )
        if no_info not in ("mean", "raise"):
            raise ValueError(f"no_info must be 'mean' or 'raise', got {no_info!r}.")
        K = self.max_score
        x = np.array(self.responses, dtype=np.float64)
        est = item_locations.reindex(self.responses.columns).values.astype(float)
        loc = est[:, None] - est[None, :]
        counts = np.array([(x == j).sum() for j in range(K + 1)])
        empty = [j for j in range(1, K) if counts[j] == 0]

        links = []
        for k in range(1, K):
            num, den = self._cpat_counts(k, x)
            total = num + den
            links.append(dict(
                num=num, den=den, observations=float(total.sum()),
                mean_pair_total=float(total[total > 0].mean()) if (total > 0).any() else 1.0,
                affected=bool(set(empty) & {k - 1, k, k + 1}),
            ))

        def plain(link, c):
            if threshold_method == "cpat-cml" and link["observations"] >= self.CPAT_THIN_COUNT:
                return self._cpat_cml_distance(link["num"], link["den"], loc, c)
            return self._cpat_info_distance(link["num"], link["den"], loc, c)

        lam = None
        if empty:
            # Category width from the distances not touching an empty
            # category; failing that, from a fit with the empty ones merged.
            first = [abs(plain(l, constant)) for l in links if not l["affected"]]
            first = [w for w in first if np.isfinite(w)]
            if first:
                width = float(np.mean(first))
            else:
                width = self._COLLAPSED_WIDTH_FACTOR * self._collapsed_category_width(empty)
            a, b, lo, hi, fallback = self._EMPTY_CATEGORY_RULES[threshold_method]
            if np.isfinite(width) and width > 0:
                lam = float(np.clip(a * width ** b, lo, hi))
            else:
                lam = fallback

        distances, thin, no_information = [], [], []
        for k, link in enumerate(links, start=1):
            if link["affected"]:
                c = lam * link["mean_pair_total"]
                if threshold_method == "cpat-cml":
                    d = self._cpat_cml_distance(link["num"], link["den"], loc, c)
                else:
                    d = self._cpat_info_distance(link["num"], link["den"], loc, c)
            else:
                d = plain(link, constant)
                if 0 < link["observations"] < self.CPAT_THIN_COUNT:
                    thin.append(k)
            if not np.isfinite(d):
                if no_info == "raise":
                    raise ValueError(self._no_info_message(k))
                no_information.append(k)
            distances.append(d)

        distances = np.array(distances, dtype=float)
        if no_information:
            ok = np.isfinite(distances)
            distances[~ok] = distances[ok].mean() if ok.any() else 0.0
            self._warn_no_information(no_information)
        if thin:
            self._warn_thin_distances(
                thin,
                " {these} estimated with the information-weighted estimator, which is more stable "
                "than 'cpat-cml' on so few observations." if threshold_method == "cpat-cml" else "",
            )
        if empty:
            self._warn_empty_categories(empty)

        thresholds = np.concatenate([[0.0], np.cumsum(distances)])
        thresholds -= thresholds.mean()
        return thresholds

    def _warn_no_information(self, no_information):
        """Warn that the distances in no_information (k: between thresholds k
        and k+1) have no comparison patterns and were set to a placeholder."""
        pairs = ", ".join(f"{k} and {k + 1}" for k in no_information)
        many = len(no_information) > 1
        warnings.warn(
            f"The data contain no information on the distance{'s' if many else ''} "
            f"between thresholds {pairs}: no pair of different items has any person "
            f"scoring k on both, or k-1 on one and k+1 on the other. The "
            f"distance{'s have' if many else ' has'} been set to the mean of the "
            f"other distances (0 if there are none), as a placeholder; the thresholds "
            f"involved are not estimated by the data. Options: collapse "
            f"sparsely used categories, or collect a larger or better-"
            f"targeted sample. Pass no_info='raise' to stop instead.",
            UserWarning,
            stacklevel=4,
        )

    def _warn_thin_distances(self, thin, note=""):
        """Warn that the distances in thin are poorly informed. note may
        contain '{these}', filled with 'These distances were'/'This distance
        was'."""
        pairs = ", ".join(f"{k} and {k + 1}" for k in thin)
        many = len(thin) > 1
        note = note.format(these="These distances were" if many else "This distance was")
        warnings.warn(
            f"The distance{'s' if many else ''} between thresholds {pairs} "
            f"{'are' if many else 'is'} poorly informed: "
            f"{'their' if many else 'its'} comparison patterns (k on two different items, or k-1 on "
            f"one and k+1 on another) are observed fewer than "
            f"{self.CPAT_THIN_COUNT} times in total.{note} Treat these "
            f"thresholds with caution. Options: collapse sparsely used "
            f"categories, or collect a larger or better-targeted sample.",
            UserWarning,
            stacklevel=4,
        )

    def _warn_empty_categories(self, empty):
        cats = ", ".join(str(j) for j in empty)
        plural = len(empty) > 1
        if self.max_score == 2:
            advice = (
                "With max_score=2, every observed response is 0 or 2, so the "
                "items are effectively dichotomous, scored 0/2; collapsing the "
                "category would leave nothing to estimate. Consider whether "
                "category 1 is a genuine response option. If not, the items "
                "can be analysed as dichotomous (SLM) after recoding 2 to 1, "
                "which changes the unit of the scale."
            )
        else:
            advice = (
                "If the categor" + ("ies were" if plural else "y was") + " never a "
                "genuine response option, or the scores may be miscoded, "
                "collapse or rescore (shift the scores above it down by one "
                "and reduce max_score). Otherwise the estimates stand, and a "
                "larger or better-targeted sample may observe the "
                "categor" + ("ies." if plural else "y.")
            )
        warnings.warn(
            f"Score categor{'ies' if plural else 'y'} {cats} {'have' if plural else 'has'} "
            f"no responses. Under the rating scale model an empty category "
            f"implies that the two thresholds around it are disordered, and "
            f"the data identify only the point where the category response "
            f"curves of the categories either side of it intersect, not the two "
            f"thresholds separately. The distances "
            f"touching the empty categor{'ies' if plural else 'y'} have been "
            f"estimated with extra smoothing targeted at them, which keeps the "
            f"rest of the scale intact; treat the individual thresholds around "
            f"{'each' if plural else 'the'} empty category as placeholders. "
            f"{advice}",
            UserWarning,
            stacklevel=3,
        )

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    @staticmethod
    def _default_constant(no_of_persons, no_of_items, max_score):
        """
        Data-adaptive default item-stage smoothing constant for
        calibrate(constant=None): SLM's rule with N replaced by N * max_score,
        min(1.0, 0.548 * I ** (-0.2 * log10(N * K))), since RSM's pairwise
        counts pool the max_score adjacent-category comparisons. Chosen by
        simulation on person-measurement error and validated out of sample.
        """
        return min(1.0, 0.548 * no_of_items ** (-0.2 * log10(max(no_of_persons * max_score, 1))))

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
        self, constant=None, threshold_constant=None, method=None,
        matrix_power=None, log_lik_tol=None, threshold_method=None,
        no_info="mean", estimation=None):
        """
        Calibrate item locations and the shared Rasch-Andrich thresholds.

        estimation='cp-full-rcl' (default): regularised composite likelihood on all
        response-pattern pairs. For two items, every pair of response
        patterns (a, b) and (a-1, b+1) with the same total is a
        binomial comparison whose log-odds depend only on the two items'
        locations and two thresholds (the person cancels). All of them are
        fitted jointly, as one composite likelihood, by Newton-Raphson.
        Additive smoothing constants go on the item comparisons and on
        CPAT's threshold patterns only; the other patterns are left
        unsmoothed (pulling them towards even odds would compress the
        thresholds). These other patterns are most informative for item
        pairs far apart in difficulty, where CPAT's adjacent patterns are
        lopsided.

        estimation='cpat-rcl': the same composite likelihood restricted to
        CPAT's patterns: odd totals give the item comparisons (persons
        scoring one point higher on one item than another), even totals the
        threshold patterns ((k, k) against (k-1, k+1) on two items).

        Both RCL methods use additive smoothing constants on the pattern
        counts as above. A category width
        with fewer than 20 threshold patterns, or with all its patterns on
        one side, is pulled toward the mean of the other widths by a ridge
        of 0.5 * (1 - T/20) (0.5 if one-sided), T its pattern count. The
        widths next to an empty middle category are not identified by
        their own patterns: they are left out of the joint fit and
        estimated by the cpat-cml rule of threshold_set(), given the item
        locations from the joint fit.

        estimation='pair-cpat': the two-stage method used before v1.4.0
        (Elliott & Buttery, 2022). Stage 1 builds a pairwise contingency
        matrix (entry (i,j) = count of persons scoring one point higher on
        item i than j), resolves structural zeroes via matrix powers, and
        extracts item locations with priority_vector(). Stage 2, given the
        item locations, estimates adjacent-threshold distances via CPAT
        (threshold_set()), chains them, and centres the result.

        Issues a UserWarning if only one item is present or if the resolved
        item constant is 0 and any item has all-maximum scores.

        Parameters
        ----------
        constant : float or None, default None
            Additive smoothing constant for the item comparisons, per item
            pair (with estimation='cp-full-rcl' (or 'cpat-rcl'), spread over the K boundaries:
            constant / K on each observed (item pair, boundary) cell).
            ``None`` resolves to the data-adaptive
            ``min(1, 0.548 * I ** (-0.2 * log10(N * K)))``, with N the number
            of persons with at least one response, I the number of items and
            K = max_score. If ``threshold_constant`` is left ``None``
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
        method : str or None, default None
            estimation='pair-cpat' only: priority vector extraction method.
            None resolves to 'log-lik'.
        matrix_power : int or None, default None
            estimation='pair-cpat' only: initial matrix power before checking
            for structural zeroes. None resolves to 0 for method='log-lik'
            (no powering), else 5.
        log_lik_tol : float or None, default None
            estimation='pair-cpat' only: log-likelihood convergence tolerance
            for priority vector extraction. None resolves to 0.000001.
        threshold_method : {'cpat-cml', 'cpat-info-weighted'} or None, default None
            estimation='pair-cpat' only: estimator for the Stage 2 threshold
            distances. None resolves to
            'cpat-cml', conditional maximum likelihood, which does not
            compress the scale; distances observed fewer than 20 times fall
            back to the information-weighted estimator. 'cpat-info-weighted'
            is the harmonic-mean information-weighted estimator of Elliott &
            Buttery (2022, eq. 4.18), used before v1.4.0, which compresses
            the scale slightly in thin data. See threshold_set().
        no_info : {'mean', 'raise'}, default 'mean'
            Handling of a threshold distance with no information at all; see
            threshold_set().
        estimation : {'cp-full-rcl', 'cpat-rcl', 'pair-cpat'} or None, default None
            Estimation method; see above. None resolves to the method of the
            object's last calibration, or 'cp-full-rcl' for the first one.
            method, matrix_power, log_lik_tol and threshold_method apply to
            'pair-cpat' only; other values passed with 'cp-full-rcl' or 'cpat-rcl'
            are ignored, with a UserWarning.

        Attributes set
        --------------
        estimation : str
            The estimation method used.
        convergence : dict or None
            estimation='cp-full-rcl' or 'cpat-rcl': Newton-Raphson iterations and
            whether the fit converged. None for 'pair-cpat'.
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
        #   constant=None                     -> item: data-adaptive (_default_constant)
        #   constant=c                        -> item c
        #   threshold_constant=None           -> threshold follows `constant`
        #                                        (0.03 if constant is None, else c)
        #   threshold_constant=t              -> threshold t (independent)
        if constant is None:
            no_of_persons = int(self.responses.notna().any(axis=1).sum())
            item_constant = self._default_constant(
                no_of_persons, self.responses.shape[1], self.max_score
            )
        else:
            item_constant = float(constant)
        if threshold_constant is not None:
            thresh_constant = float(threshold_constant)
        elif constant is None:
            thresh_constant = 0.03
        else:
            thresh_constant = float(constant)
        self.constant = item_constant
        self.threshold_constant = thresh_constant
        if estimation is None:
            estimation = getattr(self, "estimation", "cp-full-rcl")
        if estimation not in ("cp-full-rcl", "cpat-rcl", "pair-cpat"):
            raise ValueError(
                f"estimation must be 'cp-full-rcl', 'cpat-rcl', 'pair-cpat' or None, got {estimation!r}."
            )
        self.estimation = estimation
        if estimation in ("cp-full-rcl", "cpat-rcl"):
            ignored = [
                f"{name}={value!r}" for name, value, default in (
                    ("method", method, "log-lik"), ("matrix_power", matrix_power, None),
                    ("log_lik_tol", log_lik_tol, 0.000001), ("threshold_method", threshold_method, "cpat-cml"),
                ) if value is not None and value != default
            ]
            if ignored:
                warnings.warn(
                    f"{', '.join(ignored)} {'applies' if len(ignored) == 1 else 'apply'} to "
                    f"estimation='pair-cpat' only and {'is' if len(ignored) == 1 else 'are'} "
                    f"ignored with estimation='{estimation}'.",
                    UserWarning,
                    stacklevel=2,
                )
            self._calibrate_rcl(item_constant, thresh_constant, no_info, full=estimation == "cp-full-rcl")
            return
        self.convergence = None
        if method is None:
            method = "log-lik"
        if log_lik_tol is None:
            log_lik_tol = 0.000001
        if threshold_method is None:
            threshold_method = "cpat-cml"
        if threshold_method not in self._THRESHOLD_METHODS:
            raise ValueError(
                f"threshold_method must be one of {self._THRESHOLD_METHODS} or None, "
                f"got {threshold_method!r}."
            )
        self.threshold_method = threshold_method

        if item_constant == 0:
            # Unsmoothed PAIR: re-run the structural connectivity check on the
            # data as it stands now (not just at __init__(validate=True)) --
            # constant=0 removes the only thing masking a directionally
            # isolated item or disconnected component, both of which
            # check_data_connectivity() already detects and warns about
            # (does not raise; 'log-lik' tolerates an incomplete comparison
            # graph and still returns an answer for the rest of the items).
            self.check_data_connectivity()

        self.null_persons = self.responses.index[self.responses.isnull().all(axis=1)]
        if len(self.null_persons):  # drop() copies the whole table even when there is nothing to drop
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
            self.threshold_set(
                self.items, constant=thresh_constant,
                threshold_method=threshold_method, no_info=no_info,
            ),
            index=range(1, self.max_score + 1),
        )

    def _calibrate_rcl(self, item_constant, thresh_constant, no_info="mean", full=True):
        """RCL calibration (see calibrate()): items and category widths fitted jointly, on the full pattern set
        (estimation='cp-full-rcl') or CPAT's patterns ('cpat-rcl')."""
        self.threshold_method = None
        if item_constant == 0:
            self.check_data_connectivity()
        self.null_persons = self.responses.index[self.responses.isnull().all(axis=1)]
        if len(self.null_persons):  # drop() copies the whole table even when there is nothing to drop
            self.responses = self.responses.drop(self.null_persons)
        self.no_of_persons = self.responses.shape[0]
        K, I = self.max_score, self.no_of_items
        X = np.array(self.responses, dtype=np.float64)[None]
        counts = _rcl.pattern_counts(X, K)
        cat_counts = _rcl.category_counts(X, K)
        split = _rcl.empty_widths(cat_counts, K)
        rho = _rcl.width_ridges(counts[1], counts[2], K, exclude=split)
        x = X[0]
        observations = {k: float(sum(a.sum() for a in self._cpat_counts(k, x))) for k in range(1, K)}
        no_information = [k for k in range(1, K) if observations[k] == 0 and k - 1 not in split]
        thin = [k for k in range(1, K) if 0 < observations[k] < self.CPAT_THIN_COUNT and k - 1 not in split]
        if no_information and no_info == "raise":
            raise ValueError(self._no_info_message(no_information[0]))
        cm, theta, info = _rcl.fit_cell(
            "global", counts, 1, I, K, item_constant / K, thresh_constant, 0.0,
            width_rho=rho, fixed_widths={k: 0.0 for k in split},
            full=dict(X=X, empty=_rcl.empty_categories(cat_counts)) if full else None,
        )
        if no_information:
            self._warn_no_information(no_information)
        if thin:
            self._warn_thin_distances(
                thin,
                " {these} pulled toward the mean of the other distances, more strongly the "
                "fewer the observations." if K > 2 else "",
            )
        parts = cm.parts(theta)
        self.items = pd.Series(parts["delta"], index=self.responses.columns)
        widths = np.diff(parts["tau"])
        if split:
            shipped = self.threshold_set(
                self.items, constant=thresh_constant, threshold_method="cpat-cml", no_info=no_info
            )
            widths[split] = np.diff(np.asarray(shipped, dtype=float))[split]
        tau = np.concatenate([[0.0], np.cumsum(widths)])
        self.thresholds = pd.Series(tau - tau.mean(), index=range(1, K + 1))
        self._rcl_fit = (cm, theta, info)
        self.convergence = dict(iterations=info["iterations"], converged=info["converged"])
        if not info["converged"]:
            warnings.warn(
                f"RCL-CPAT calibration did not converge in {info['iterations']} "
                f"Newton-Raphson iterations; the estimates may be unreliable.",
                UserWarning,
                stacklevel=3,
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
        estimation=None,
        threshold_method=None,
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
        estimation, threshold_method : str or None, default None
            As for calibrate(), used only if calibrate is triggered. None
            resolves as in calibrate().

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
                estimation=estimation,
                threshold_method=threshold_method,
            )

        self._calibrate_anchor_core(
            anchors, selection_method, corr_tol, sd_ratio_tol, min_anchors,
            wald_alpha, no_of_samples, adj, overwrite_anchors, plot, plot_kwargs,
            seed,
            std_errors_kwargs=dict(
                constant=constant, threshold_constant=threshold_constant,
                method=method, matrix_power=matrix_power,
                log_lik_tol=log_lik_tol,
            ),
        )

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

        self._snapshot_calibrate_anchor_run(
            anchors,
            extra=dict(
                anchor_threshold_test=self.anchor_threshold_test,
                anchor_category_width_test=self.anchor_category_width_test,
            ),
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

    def _sandwich_obstacle(self):
        """None if sandwich standard errors apply; otherwise a description of
        what the threshold estimates rely on that the sandwich does not
        model (an unused category, a thin or uninformative category width,
        or the information-weighted estimator). With estimation='cp-full-rcl' (or 'cpat-rcl')
        thin widths are part of the penalised composite likelihood, so only
        an unused category is an obstacle."""
        if getattr(self, "estimation", "pair-cpat") in ("cp-full-rcl", "cpat-rcl"):
            x = np.array(self.responses, dtype=np.float64)
            unused = [c for c in range(self.max_score + 1) if (x == c).sum() == 0]
            if unused:
                return ("; ".join(f"category {c} has 0 responses across all items" for c in unused))
            if not self.convergence["converged"]:
                return "the calibration did not converge, and the sandwich assumes it did"
            cm, theta, info = self._rcl_fit
            try:
                undetermined = _rcl.cell_null(cm, theta, info).shape[1]
            except _rcl.SandwichTooLarge as ex:
                return (f"the design is too large to form the analytic standard errors' "
                        f"information matrix ({ex})")
            if undetermined:
                _rcl.release_state(info)
                return (
                    "the data do not identify some of the estimates (for example the locations "
                    "of items not linked to the others), whose bootstrap standard errors reflect "
                    "the smoothing rather than the data"
                )
            return None
        if getattr(self, "threshold_method", "cpat-cml") != "cpat-cml":
            return "threshold_method='cpat-info-weighted' was used"
        x = np.array(self.responses, dtype=np.float64)
        unused = [c for c in range(self.max_score + 1) if (x == c).sum() == 0]
        if unused:
            return ("; ".join(f"category {c} has 0 responses across all items" for c in unused))
        thin = []
        for k in range(1, self.max_score):
            num, den = self._cpat_counts(k, x)
            total = int((num + den).sum())
            if total < self.CPAT_THIN_COUNT:
                thin.append(
                    f"the distance between thresholds {k} and {k + 1} has {total} comparison "
                    f"observations (responses of {k} on two different items, or {k - 1} on "
                    f"one and {k + 1} on another; fewer than {self.CPAT_THIN_COUNT})"
                )
        if thin:
            return "; ".join(thin)
        return None

    def _sandwich_covariance(self, full=False, small_sample=False):
        """
        Sandwich covariance of (item locations, category widths) for the
        stacked estimating equations: Bradley-Terry scores of the PAIR item
        stage (adjacent-category wins, persons as units) and cpat-cml scores
        of each category width, with the item differences as offsets. The
        bread is block lower-triangular (widths depend on items, not the
        reverse); the item block has the usual one-dimensional null space
        (location origin), handled with a pseudo-inverse.
        Returns (V_items, V_widths), I x I and (K-1) x (K-1); with full=True,
        the whole (I + K - 1) square covariance of (item locations, widths).
        small_sample=True applies the Kauermann-Carroll correction to the
        stacked scores (see _kc_scores()).

        With estimation='cp-full-rcl' (or 'cpat-rcl'), the bread is the negative Hessian of the
        penalised composite log-likelihood at the estimates and the meat the
        sum over persons of their score outer products (see _rcl).
        """
        if getattr(self, "estimation", "pair-cpat") in ("cp-full-rcl", "cpat-rcl"):
            return self._rcl_sandwich_covariance(full=full, small_sample=small_sample)
        K = self.max_score
        X = np.array(self.responses, dtype=np.float64)
        N, I = X.shape
        E = [np.nan_to_num((X == c).astype(float)) for c in range(K + 1)]
        b = -self.items.reindex(self.responses.columns).values.astype(float)
        D = np.diff(self.thresholds.values.astype(float))
        off = 1.0 - np.eye(I)
        with np.errstate(over="ignore"):
            pi = 1.0 / (1.0 + np.exp(-(b[:, None] - b[None, :])))
        P1 = pi * off
        psi = np.zeros((N, I))
        wins = np.zeros((I, I))
        for c in range(K):
            hi, lo = E[c + 1], E[c]
            psi += hi * (lo @ off.T) - (hi * (lo @ P1.T) + lo * (hi @ P1.T))
            wins += hi.T @ lo
        w = (wins + wins.T) * off * pi * (1 - pi)
        L = np.diag(w.sum(axis=1)) - w
        s = np.zeros((N, K - 1))
        H = np.zeros(K - 1)
        cross = np.zeros((K - 1, I))
        for k in range(1, K):
            with np.errstate(over="ignore"):
                p = off / (1.0 + np.exp(-(D[k - 1] + (b[:, None] - b[None, :]))))
            ek, ekm, ekp = E[k], E[k - 1], E[k + 1]
            s[:, k - 1] = (ek.sum(axis=1) ** 2 - (ek ** 2).sum(axis=1)
                           - ((ek @ p) * ek).sum(axis=1) - ((ekm @ p) * ekp).sum(axis=1))
            v = (ek.T @ ek + ekm.T @ ekp) * off * p * (1 - p)
            H[k - 1] = v.sum()
            cross[k - 1] = v.sum(axis=1) - v.sum(axis=0)
        U = np.hstack([psi, s])
        Lp = self._origin_pinv(L)
        A = np.zeros((I + K - 1, I + K - 1))
        A[:I, :I] = Lp
        A[I:, I:] = np.diag(1.0 / H)
        A[I:, :I] = -np.diag(1.0 / H) @ cross @ Lp
        if small_sample:
            v_items = off * pi * (1 - pi)
            v_widths = []
            for k in range(1, K):
                with np.errstate(over="ignore"):
                    p = off / (1.0 + np.exp(-(D[k - 1] + (b[:, None] - b[None, :]))))
                v_widths.append(off * p * (1 - p))

            def person_bread(sl):
                # rows/columns: item strengths, then widths; the bread is
                # block lower-triangular, as above
                n = sl.stop - sl.start
                Bn = np.zeros((n, I + K - 1, I + K - 1))
                Bn[:, :I, :I] = self._bt_person_curvature(
                    [(E[c + 1][sl], E[c][sl]) for c in range(K)], v_items)
                for k in range(1, K):
                    ek, ekm, ekp = E[k][sl], E[k - 1][sl], E[k + 1][sl]
                    vn = (ek[:, :, None] * ek[:, None, :]
                          + ekm[:, :, None] * ekp[:, None, :]) * v_widths[k - 1][None]
                    Bn[:, I + k - 1, I + k - 1] = vn.sum(axis=(1, 2))
                    Bn[:, I + k - 1, :I] = vn.sum(axis=2) - vn.sum(axis=1)
                return Bn

            U = self._kc_scores(U, person_bread, A)
        V = A @ (U.T @ U) @ A.T
        if full:
            # computed in Bradley-Terry strengths b = -delta: flip the item
            # block's sign so the item-width cross-covariance refers to item
            # locations (the diagonal blocks are unaffected)
            sgn = np.concatenate([-np.ones(I), np.ones(K - 1)])
            return V * np.outer(sgn, sgn)
        return V[:I, :I], V[I:, I:]

    def _rcl_sandwich_covariance(self, full=False, small_sample=False):
        """RCL-CPAT counterpart of _sandwich_covariance() (same return values)."""
        cm, theta, info = self._rcl_fit
        K, I = self.max_score, self.no_of_items
        X = np.array(self.responses, dtype=np.float64)[None]
        V, free, null = _rcl.cell_sandwich(cm, theta, info, X, K, small_sample=small_sample,
                                           kc_scores=self._kc_scores)
        _rcl.release_state(info)
        Vfull = np.full((cm.p, cm.p), np.nan)
        Vfull[np.ix_(free, free)] = V
        sd, st = cm.slices["delta"], cm.slices["tau"]
        Mmap = np.zeros((I + K - 1, cm.p))
        Mmap[:I, sd] = cm._NI
        Mmap[I:, st] = np.eye(K - 1)
        Vout = Mmap @ np.nan_to_num(Vfull) @ Mmap.T
        if null.shape[1]:
            # directions the data do not identify: no standard error
            hit = np.abs(Mmap[:, free] @ null).max(axis=1) > 1e-8
            Vout[hit, :] = np.nan
            Vout[:, hit] = np.nan
            _rcl.warn_unidentified(stacklevel=4)
        if (~free).any():
            fixed = I + np.nonzero(~free[st])[0]
            Vout[fixed, :] = np.nan
            Vout[:, fixed] = np.nan
        if full:
            return Vout
        return Vout[:I, :I], Vout[I:, I:]

    def _sandwich_std_errors(self, interval):
        """Set the std_errors() attributes from the sandwich covariance."""
        K = self.max_score
        V_items, V_widths = self._sandwich_covariance()
        cols = self.responses.columns
        M = np.vstack([np.zeros(K - 1), np.tril(np.ones((K - 1, K - 1)))])
        M = M - M.mean(axis=0)  # thresholds = centred cumulative widths
        self.item_se = pd.Series(np.sqrt(np.clip(np.diag(V_items), 0, None)), index=cols)
        self.threshold_se = np.sqrt(np.clip(np.diag(M @ V_widths @ M.T), 0, None))
        width_se = np.sqrt(np.clip(np.diag(V_widths), 0, None))
        self.cat_width_se = pd.Series(width_se, index=range(1, K))
        self.se_covariance = dict(items=pd.DataFrame(V_items, index=cols, columns=cols),
                                  cat_widths=pd.DataFrame(V_widths, index=range(1, K), columns=range(1, K)))
        self.item_bootstrap = self.threshold_bootstrap = self.cat_width_bootstrap = None
        if interval is not None:
            z = norm.ppf(0.5 + interval / 2)
            items = self.items.reindex(cols)
            self.item_low, self.item_high = items - z * self.item_se, items + z * self.item_se
            thr = self.thresholds.values.astype(float)
            self.threshold_low, self.threshold_high = thr - z * self.threshold_se, thr + z * self.threshold_se
            widths = np.diff(thr)
            self.cat_width_low = {k: widths[k - 1] - z * width_se[k - 1] for k in range(1, K)}
            self.cat_width_high = {k: widths[k - 1] + z * width_se[k - 1] for k in range(1, K)}
        else:
            self.item_low = self.item_high = None
            self.threshold_low = self.threshold_high = None
            self.cat_width_low = self.cat_width_high = None

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
        se_method=None,
    ):
        """
        Estimate standard errors for item locations, thresholds and category widths.

        se_method='sandwich' computes analytic standard errors from the
        calibrated estimates: a sandwich (robust) covariance for the stacked
        estimating equations of both stages -- the Bradley-Terry equations of
        the PAIR item stage and the cpat-cml equations of the threshold
        stage -- with persons as the independent units. Stacking carries the
        item-location uncertainty into the thresholds. In simulation its
        standard errors were within a few percent of the true sampling
        standard deviation (and of the bootstrap) for items and thresholds,
        at about the cost of one calibration.

        se_method='bootstrap' draws no_of_samples bootstrap resamples of
        person-level response data, calibrates each, and computes SDs of the
        estimates across samples.

        se_method=None (default) uses the sandwich unless the threshold
        estimates rely on something it does not model -- an unused score
        category (targeted smoothing); with estimation='pair-cpat', also a
        poorly informed or uninformative category width (information-
        weighted fallback or placeholder), or threshold_method=
        'cpat-info-weighted' -- or, with estimation='cp-full-rcl' (or 'cpat-rcl'), the
        calibration did not converge or the data do not identify some
        estimates; in which case it uses the bootstrap, with a UserWarning
        giving the reason. So standard errors are always produced.
        self.se_method records the method used.

        Parameters
        ----------
        interval : float or None, default None
            CI width (e.g. 0.95). If None, only SEs computed.
        no_of_samples : int, default 500
            Number of bootstrap resamples.
        constant : float or None, default None
            Smoothing constant for bootstrap calibrations, passed to
            calibrate(); None resolves to the data-adaptive default (see calibrate()).
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
        se_method : {None, 'sandwich', 'bootstrap'}, default None
            Standard error method; see above. With 'sandwich', confidence
            intervals are estimate +/- z * SE, and the bootstrap attributes
            are None.

        Attributes set
        --------------
        se_method : str
            'sandwich' or 'bootstrap', the method actually used.
        item_se : pandas.Series
            SE for each item location.
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
            Bootstrap category width estimates (None with the sandwich).
        """
        if se_method not in (None, "sandwich", "bootstrap"):
            raise ValueError(
                f"se_method must be None, 'sandwich' or 'bootstrap', got {se_method!r}."
            )
        if not hasattr(self, "thresholds"):
            self.calibrate(
                constant=constant, threshold_constant=threshold_constant, method=method,
                matrix_power=matrix_power, log_lik_tol=log_lik_tol,
            )
        if se_method is None:
            obstacle = self._sandwich_obstacle()
            se_method = "sandwich" if obstacle is None else "bootstrap"
            if obstacle is not None:
                sparse = not obstacle.startswith(("the calibration", "the data do not"))
                warnings.warn(
                    f"Standard errors computed by bootstrap rather than analytically "
                    f"(sandwich), because {obstacle}"
                    + (": the threshold estimates then rely on smoothing or a fallback "
                       "estimator that the analytic standard errors do not account for. This "
                       "usually indicates sparse data in some categories; consider collapsing "
                       "sparsely used categories or collecting more data." if sparse else ".")
                    + " Pass se_method= to choose the method explicitly.",
                    UserWarning,
                    stacklevel=2,
                )
        self.se_method = se_method
        if se_method == "sandwich":
            self._sandwich_std_errors(interval)
            return

        rng = np.random.default_rng(seed)
        samples = []
        collector = _BootstrapWarnings()
        for _ in range(no_of_samples):
            with collector.resample():
                sample = RSM(
                    self.responses.sample(frac=1, replace=True, random_state=rng),
                    max_score=self.max_score,
                )
                sample.calibrate(estimation=getattr(self, "estimation", None),
                    constant=constant,
                    threshold_constant=threshold_constant,
                    threshold_method=getattr(self, "threshold_method", "cpat-cml"),
                    method=method,
                    matrix_power=matrix_power,
                    log_lik_tol=log_lik_tol,
                )
            samples.append(sample)
        collector.emit()

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
            If True, returns Warm's (1989) weighted likelihood estimate, found by
            solving the weighted likelihood equation directly (bracketed
            Newton). If False, returns the plain ML estimate.
        tolerance : float, default 0.00001
            Convergence criterion per person.
        max_iters : int, default 100
            Maximum iterations. Non-converged persons set to NaN.
        ext_score_adjustment : float, default 0.5
            Amount added to a score of 0 (or subtracted from a maximum score)
            for plain ML (warm_corr=False), which has no finite estimate for
            extreme scores; also sets the starting value. Has no effect on
            Warm estimates, which are finite for extreme scores.

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
                pf_arr = person_filter.loc[estimates.index].values
                estimates = pd.Series(
                    self._solve_wle(
                        person_data.loc[estimates.index].sum(axis=1).values,
                        estimates.values,
                        lambda t, rows: self._wle_moments(t, item_location_arr, pf_arr[rows]),
                        tolerance,
                        max_iters,
                    ),
                    index=estimates.index,
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
            item_location_arr = item_locations.values
            pf_arr = np.ones((1, len(items)))
            estimate = float(
                self._solve_wle(
                    [float(score)],
                    [estimate],
                    lambda t, rows: self._wle_moments(t, item_location_arr, pf_arr[rows]),
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
        filename : str or None, default None
            If given, also saves the returned table to
            ``{filename}.{file_format}`` (path without extension, as for
            the plotting methods).
        file_format : {'csv', 'xlsx'}, default 'csv'
            File format used when filename is given.

        Attributes set
        --------------
        person_table : pandas.Series
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
            pf_arr = np.ones((len(scores), no_of_items))
            estimates = pd.Series(
                self._solve_wle(
                    scores,
                    estimates.values,
                    lambda t, rows: self._wle_moments(t, item_location_arr, pf_arr[rows]),
                    tolerance,
                    max_iters,
                ),
                index=scores,
            )

        self.score_table = estimates
        _save_table(self.score_table, filename, file_format)
        return self.score_table

    def _wle_moments(self, person_locations, item_locations, person_filter):
        """
        Expected score, information, and third and fourth cumulants of the
        total score at the given person locations, for _solve_wle().

        person_locations : array (N,); item_locations : array (I,);
        person_filter : array (N, I), 1 = observed, 0 or NaN = missing.
        """
        probs, cats_arr = self._cat_probs_matrix(
            np.asarray(person_locations, dtype=float), item_locations, self.thresholds
        )
        return self._polytomous_moments(probs, cats_arr, person_filter)

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

    def category_counts_df(
        self, persons=None, items=None, counts_name=None, filename=None, file_format="csv"
    ):
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
        filename : str or None, default None
            If given, also saves the returned table to
            ``{filename}.{file_format}`` (path without extension, as for
            the plotting methods).
        file_format : {'csv', 'xlsx'}, default 'csv'
            File format used when filename is given.

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

        _save_table(df, filename, file_format)
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
            Smoothing constant, passed to calibrate(); None resolves to the data-adaptive default (see calibrate()).
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

    def _global_fit_subset(self, idx):
        """Calibrated RSM on a subset of persons (global_fit())."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = RSM(self.responses.loc[idx], max_score=self.max_score, validate=False)
            m.calibrate(estimation=getattr(self, "estimation", None), threshold_method=getattr(self, "threshold_method", "cpat-cml"))
        return m

    @staticmethod
    def _global_fit_params(m, small_sample=False):
        """(labels, estimates, covariance, comparable mask, location blocks) for
        global_fit(): item locations (centred) and category widths, with the
        full stacked sandwich covariance. Widths needing a fallback in this
        group (an unused category next to them, or fewer than CPAT_THIN_COUNT
        comparison observations) are marked not comparable."""
        names = list(m.responses.columns)
        I, K = len(names), m.max_score
        items = m.items.reindex(names).values.astype(float)
        est = np.concatenate([items - items.mean(), np.diff(m.thresholds.values.astype(float))])
        V = m._sandwich_covariance(full=True, small_sample=small_sample)
        x = np.array(m.responses, dtype=np.float64)
        ok = np.ones(I + K - 1, bool)
        # an item scored the same by everyone in this group is not estimable here
        ok[:I] = [len(np.unique(x[~np.isnan(x[:, i]), i])) > 1 for i in range(I)]
        for k in range(1, K):
            num, den = m._cpat_counts(k, x)
            empty_near = any((x == c).sum() == 0 for c in (k - 1, k, k + 1))
            ok[I + k - 1] = (not empty_near) and (num + den).sum() >= m.CPAT_THIN_COUNT
        labels = names + [f"Category width {k}" for k in range(1, K)]
        return labels, est, V, ok, [np.arange(I)]

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
        warnings.warn(
            "andersen_lr_test() is deprecated; use global_fit() (Wald), which is "
            "calibrated for any split, including split='exogenous' with "
            "covariate=. global_fit(method='andersen-lr') is reserved for "
            "conditional maximum likelihood estimates (planned).",
            FutureWarning,
            stacklevel=2,
        )
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
        m_full.calibrate(estimation=getattr(self, "estimation", None),
            constant=constant,
            threshold_constant=threshold_constant,
            threshold_method=getattr(self, "threshold_method", "cpat-cml"),
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
            m.calibrate(estimation=getattr(self, "estimation", None),
                constant=constant,
                threshold_constant=threshold_constant,
                threshold_method=getattr(self, "threshold_method", "cpat-cml"),
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
        dcf_cap=80000,
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
            guard against the opposite problem from welch's: at very
            large N, SE shrinks enough that trivial differences become
            "significant", swamping the logit-magnitude thresholds'
            intent. The two components use different mechanisms, because
            they were found to need them (see reference_n/dcf_cap below):
            item-location SE is rescaled by sqrt(actual_n / reference_n)
            unconditionally (Tristán, 2006, Rasch Measurement Transactions
            20:3); threshold (DCF) SE is rescaled by a per-threshold
            dynamic cap on its own CPAT total_weight (dcf_cap), applied
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
            fixed-reference_n approach this package used for threshold
            (DCF) SE before dcf_cap replaced it (see dcf_cap).
        dcf_cap : float, default 80000
            Cap, in CPAT total_weight units (NOT a person count, unlike
            reference_n), for size_adjust=True's threshold (DCF) component
            only — does not affect dif_table. Per threshold, independently
            for each group: effective_weight = min(total_weight, dcf_cap);
            se_adjusted = se_actual * max(1, sqrt(total_weight/dcf_cap)),
            where total_weight is that threshold's own CPAT harmonic-mean
            information weight, summed across every item pair (see
            _threshold_total_weights — the same weighting _threshold_distance
            itself uses, Elliott & Buttery 2022). Below the cap, no
            adjustment at all — unlike reference_n's unconditional rescale,
            an ordinary-sized study's own SE is used as-is. This replaces
            an earlier design (a single fixed reference_n shared with
            dif_table) that was found by direct simulation to make DCF
            power collapse at real effect sizes as N grows, because
            threshold information doesn't scale with raw person count the
            way item-location information does, and no single fixed
            reference_n could serve both. Calibrated by simulation across
            max_score in {3, 5, 8} (I=15, 0.8-logit category-width shift,
            N in {500,...,6000}/group) — total_weight runs roughly 5-6x
            larger than PCM's own analogous per-cell weight at matched N
            (RSM pools information across every item into one shared
            vector), so dcf_cap is correspondingly larger than PCM's own
            default. Power-saturation and Type-I-nominal cap ranges were
            stable across all three max_score values tested at matched N —
            see the raschpy_dynamic_n_calibration project notes for the
            full simulation record. Not yet recalibrated for designs far
            outside that swept range — 80000 is a reasonable, validated
            default, not a provably-universal one.
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

        self._dif_validate(
            correction, selection_method, test, category_thresholds, reference_n,
            dcf_cap=dcf_cap,
        )

        # Non-extreme persons, matching andersen_lr_test's convention
        scores = self.responses.sum(axis=1)
        max_scores = self.responses.notna().sum(axis=1) * self.max_score
        non_extreme = self.responses.index[(scores > 0) & (scores < max_scores)]

        cov_values, reference, focal_levels, group_sizes, n_missing = self._dif_group_setup(
            non_extreme, covariate, reference
        )

        needs_ll = test in ("lr", "both") or omnibus
        pe_kw = dict(
            warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
            ext_score_adjustment=ext_score_adjustment,
        )

        ref_idx = cov_values.index[cov_values == reference]
        ref_model = RSM(self.responses.loc[ref_idx], max_score=self.max_score)
        ref_model.calibrate(estimation=getattr(self, "estimation", None),
            constant=constant, threshold_constant=threshold_constant, threshold_method=getattr(self, "threshold_method", "cpat-cml"), method=method, matrix_power=matrix_power,
            log_lik_tol=log_lik_tol,
        )
        ref_model.std_errors(
            no_of_samples=no_of_samples, constant=constant, threshold_constant=threshold_constant, method=method,
            matrix_power=matrix_power, log_lik_tol=log_lik_tol, seed=seed,
        )
        if needs_ll:
            ref_model.person_estimates(**pe_kw)
            ll_ref = ref_model._log_likelihood()
        ref_total_weights = (
            ref_model._threshold_total_weights(constant=ref_model.threshold_constant)
            if size_adjust else None
        )

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
            focal_model.calibrate(estimation=getattr(self, "estimation", None),
                constant=constant, threshold_constant=threshold_constant, threshold_method=getattr(self, "threshold_method", "cpat-cml"), method=method, matrix_power=matrix_power,
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
            wald_result = self._dif_item_wald_block(
                ref_model, focal_model, ref_model.item_bootstrap, focal_model.item_bootstrap,
                selection_method, corr_tol, sd_ratio_tol, min_anchors, wald_alpha,
                size_adjust, reference_n, welch, correction, alpha, logit_threshold,
                category, category_thresholds, category_alpha,
                ref_cov=ref_model.se_covariance['items'] if ref_model.se_method == 'sandwich' else None,
                focal_cov=focal_model.se_covariance['items'] if focal_model.se_method == 'sandwich' else None,
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
                m_full = RSM(self.responses.loc[combined_idx], max_score=self.max_score)
                m_full.calibrate(estimation=getattr(self, "estimation", None),
                    constant=constant, threshold_constant=threshold_constant, threshold_method=getattr(self, "threshold_method", "cpat-cml"), method=method, matrix_power=matrix_power,
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
                    pooled_focal_scale = pooled - tc[item]

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

                lr_table = self._dif_lr_table(
                    lr_rows, common_items, correction, alpha, diff, logit_threshold
                )

            item_table = self._dif_item_table(
                focal, ref_model, focal_model, common_items, focal_shifted,
                diff, se, z, p, p_corrected, selected, flagged,
                welch_df=item_welch_df if welch else None,
                category_col=item_category_col if category else None,
                lr_table=lr_table if test in ("lr", "both") else None,
            )

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
                # DCF dynamic cap: per-threshold CPAT total_weight (not raw
                # person count) is the effective n here, capped rather than
                # pinned -- see dcf_cap docstring for why a fixed reference_n
                # (as used for item-location DIF above) was found to make
                # DCF power collapse at real sample sizes instead.
                focal_total_weights = focal_model._threshold_total_weights(
                    constant=focal_model.threshold_constant
                )
                ref_tw = ref_total_weights.reindex(thr_index).values
                focal_tw = focal_total_weights.reindex(thr_index).values
                ref_eff = np.minimum(ref_tw, dcf_cap)
                focal_eff = np.minimum(focal_tw, dcf_cap)
                ref_se_thr = pd.Series(
                    ref_se_thr.reindex(thr_index).values * np.sqrt(ref_tw / ref_eff),
                    index=thr_index,
                )
                focal_se_thr = pd.Series(
                    focal_se_thr.reindex(thr_index).values * np.sqrt(focal_tw / focal_eff),
                    index=thr_index,
                )
                ref_n_thr = ref_eff
                focal_n_thr = focal_eff

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
        pcm.calibrate(constant=constant, method=method, log_lik_tol=log_lik_tol)
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
            Smoothing constant, passed to calibrate(); None resolves to the data-adaptive default (see calibrate()).
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
        filename=None,
        file_format="csv",
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
            Smoothing constant, passed to calibrate(); None resolves to the data-adaptive default (see calibrate()).
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
        filename : str or None, default None
            If given, also saves the returned table to
            ``{filename}.{file_format}`` (path without extension, as for
            the plotting methods).
        file_format : {'csv', 'xlsx'}, default 'csv'
            File format used when filename is given.

        Attributes set
        --------------
        item_stats : pandas.DataFrame
            Item statistics with items as rows. Always contains Estimate,
            Count, Facility, Infit MS, Outfit MS. Optional: SE and CI
            bounds (if se=True).

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
        _save_table(self.item_stats, filename, file_format)
        return self.item_stats

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
        filename=None,
        file_format="csv",
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
            Smoothing constant, passed to calibrate(); None resolves to the data-adaptive default (see calibrate()).
        threshold_constant : float or None, default None
            Passed to calibrate() as the Stage 2 CPAT constant; None
            follows `constant` (0.03 when constant is None). See calibrate().
        no_of_samples : int, default 500
            Bootstrap samples.
        interval : float or None, default None
            CI width.
        filename : str or None, default None
            If given, also saves the returned table to
            ``{filename}.{file_format}`` (path without extension, as for
            the plotting methods).
        file_format : {'csv', 'xlsx'}, default 'csv'
            File format used when filename is given.

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
        _save_table(self.threshold_stats, filename, file_format)
        return self.threshold_stats

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
        filename=None,
        file_format="csv",
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
        filename : str or None, default None
            If given, also saves the returned table to
            ``{filename}.{file_format}`` (path without extension, as for
            the plotting methods).
        file_format : {'csv', 'xlsx'}, default 'csv'
            File format used when filename is given.

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

        Returns
        -------
        pandas.DataFrame
            The table, also stored as the attribute listed above.
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
        if getattr(self, "cat_width_bootstrap", None) is not None:
            prop = (self.cat_width_bootstrap < 0).mean(axis=0).values
        else:  # sandwich SEs: normal-approximation probability of a negative width
            prop = norm.cdf(-cat_widths.values / self.cat_width_se.values)
        stats["Prop disordered"] = prop.round(dp)
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
        threshold_constant=None,
        filename=None,
        file_format="csv",
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
            Smoothing constant, passed to calibrate(); None resolves to the data-adaptive default (see calibrate()).
        threshold_constant : float or None, default None
            Passed to calibrate() as the Stage 2 CPAT constant; None
            follows `constant` (0.03 when constant is None). See calibrate().
        filename : str or None, default None
            If given, also saves the returned table to
            ``{filename}.{file_format}`` (path without extension, as for
            the plotting methods).
        file_format : {'csv', 'xlsx'}, default 'csv'
            File format used when filename is given.

        Attributes set
        --------------
        person_stats : pandas.DataFrame
            Person statistics with persons as rows. Contains Estimate, CSEM,
            Score, Max score, p, Infit MS, Infit Z, Outfit MS, Outfit Z.
            Optional: RSEM.

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
        threshold_constant=None,
        alpha=False,
        seed=None,
        filename=None,
        file_format="csv",
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
            Smoothing constant, passed to calibrate(); None resolves to the data-adaptive default (see calibrate()).
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
        filename : str or None, default None
            If given, also saves the returned table to
            ``{filename}.{file_format}`` (path without extension, as for
            the plotting methods).
        file_format : {'csv', 'xlsx'}, default 'csv'
            File format used when filename is given.

        Attributes set
        --------------
        test_stats : pandas.DataFrame
            Two-column table (Items, Persons) with rows:
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
            Smoothing constant, passed to calibrate(); None resolves to the data-adaptive default (see calibrate()).
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
            Smoothing constant, passed to calibrate(); None resolves to the data-adaptive default (see calibrate()).
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
        dpi=300,
        plot_density=None,
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
        if plot_density is not None:
            warnings.warn(
                "plot_density= is deprecated and will be removed in a future version; use dpi= instead.",
                FutureWarning, stacklevel=2,
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
    def _category_widths(thresholds):
        """
        Category widths (tau_{k+1} - tau_k) from a threshold Series indexed
        1..max_score, relabelled 1..max_score-1 to match cat_width_se's
        own indexing convention (std_errors(), rsm.py).
        """
        widths = thresholds.diff().dropna()
        widths.index = range(1, len(thresholds))
        return widths

    def recovery_stats(
        self, anchor=False, alpha=0.05, correction="bh", divergence_tol=0.1,
        plot=True, **plot_kwargs,
    ):
        """
        Compare recovered parameter estimates against the true generating
        values from the simulation this object was constructed from.

        No-op (warns, returns None) unless this object was built from a
        simulation (self.generating exists) -- NOT restricted to
        self.generating.model == 'RSM'. Person recovery is valid across
        any appropriate model pairing (e.g. this RSM's persons compared
        against pseudo-estimates built from PCM_Sim-generated data), so
        it's always attempted. Item recovery is always attempted too
        (item difficulties are comparable across SLM/PCM/RSM regardless
        of threshold structure).

        The RSM<->PCM pairing specifically is stricter than a per-
        parameter-type skip: if the generating side is a PCM_Sim, NOTHING
        fires at all (the whole method no-ops, not just thresholds)
        unless EVERY max score matches -- PCM_Sim's own max_score_vector
        must be constant across items AND equal this RSM's own max_score.
        A partial mismatch means the two models aren't testing the same
        category structure at all, not merely a threshold-shape mismatch,
        so allowing items/persons through while just skipping thresholds
        would be misleading. When every max score does match, thresholds/
        category_widths are compared against the mean of PCM's per-item
        centred thresholds (all rows should already be identical/near-
        identical if the true structure is genuinely RSM-nested). Against
        SLM-generated data (no thresholds at all), thresholds/
        category_widths are skipped with a warning
        (self.recovery.thresholds/.category_widths left as None,
        items/persons still proceed) rather than raising or fabricating a
        meaningless comparison -- unlike the PCM case above, SLM doesn't
        trigger a whole-method no-op, since SLM's items are still directly
        comparable regardless of its lack of thresholds.

        Person recovery does NOT compare self.persons directly against
        self.generating.persons -- that would conflate item/threshold
        estimation and model-specification error with the irreducible
        stochastic noise of a single finite response pattern (even a
        perfect model, perfectly calibrated, shows nonzero residuals
        against true theta, purely from response-pattern discreteness).
        Instead, following Elliott & Buttery (2022a), a pseudo-estimate is
        computed: the same Newton-Raphson person() procedure re-run on the
        same self.responses, but using the TRUE generating item locations
        and thresholds in place of the estimated ones. The persons table's
        'error' column is (recovered estimate - pseudo-estimate), not
        (recovered estimate - true theta) -- both share the same actual
        response-pattern noise, so differencing cancels that shared
        stochastic component. What remains is the COMBINED effect of (a)
        item/threshold parameter estimation error and (b) any model
        misspecification (fitting a model form that doesn't match how the
        data were actually generated) on person measurement -- this method
        does not, and cannot, separate those two; Elliott & Buttery (2022a)
        used exactly this construction to demonstrate the person-
        measurement impact of (b), not just (a) -- e.g. fitting a model
        without threshold effects to data generated with them. Its
        'generating' column holds this pseudo-estimate, not the raw true
        theta -- unlike every other parameter type's 'generating' column,
        which does hold the actual true value.

        Parameters
        ----------
        anchor : bool, default False
            If True, compare self.anchor_items (from calibrate_anchor())
            against self.generating.items, instead of self.items. Raises
            ValueError if calibrate_anchor() has not been run. RSM's
            shared threshold vector is never anchored by calibrate_anchor
            (only item locations translate), so thresholds/category_widths
            and person recovery are unaffected by this flag either way.
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
            Prop_flagged). Also sets
            self.recovery.parameter_estimation_residual (float, or None
            when thresholds/category_widths are skipped): the Elliott &
            Buttery (2022) expected-score-space RMS residual between
            generating and estimated item location + shared thresholds at
            the true generating theta -- see _parameter_estimation_residual.
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
        if gen_model == "PCM":
            # Cross-model RSM<->PCM comparison: nothing fires at all
            # (not just thresholds) unless EVERY max score matches --
            # PCM's own per-item max scores must be constant AND equal
            # this RSM object's own max_score. A partial mismatch means
            # the two models aren't testing the same category structure
            # at all, not just a threshold-shape mismatch.
            msv = self.generating.max_score_vector
            if len(set(msv)) != 1 or list(set(msv))[0] != self.max_score:
                warnings.warn(
                    "recovery_stats() is a no-op: RSM<->PCM comparison "
                    "requires every max score to match -- PCM_Sim's "
                    f"max_score_vector={list(msv)} must be constant and "
                    f"equal this RSM's max_score={self.max_score}.",
                    UserWarning,
                    stacklevel=2,
                )
                return None
        elif gen_model == "SLM" and self.max_score != 1:
            # SLM_Sim's data is genuinely dichotomous -- only a valid
            # nested comparison when this RSM object's own max_score is
            # also 1 (same "every max score must match" rule as the PCM
            # case above, SLM implicitly being max_score=1).
            warnings.warn(
                "recovery_stats() is a no-op: SLM_Sim-generated data is "
                "dichotomous, only a valid nested comparison when this "
                f"RSM's own max_score is also 1 (got {self.max_score}).",
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
            self.generating.items, est_items, se=self.item_se, alpha=alpha,
            correction=correction, divergence_tol=divergence_tol,
        )
        self.recovery.items = _SimParams()
        self.recovery.items.table = items_table
        self.recovery.items.summary = items_summary

        gen_thresholds = None
        if gen_model == "RSM":
            gen_thresholds = self.generating.thresholds
        elif gen_model == "PCM":
            # Max-score match already verified above -- safe to proceed.
            gen_thresholds = self.generating.thresholds.mean(axis=0)
            gen_thresholds.index = range(1, self.max_score + 1)
        else:
            warnings.warn(
                f"recovery_stats(): thresholds/category_widths skipped "
                f"-- generating model {gen_model!r} has no threshold "
                f"structure comparable to RSM's.",
                UserWarning,
                stacklevel=2,
            )

        self.recovery.thresholds = None
        self.recovery.category_widths = None
        gen_widths = est_widths = None
        if gen_thresholds is not None:
            threshold_se = pd.Series(
                self.threshold_se, index=range(1, self.max_score + 1)
            )
            thresholds_table, thresholds_summary = self._parameter_recovery_table(
                gen_thresholds, self.thresholds, se=threshold_se,
                alpha=alpha, correction=correction,
                divergence_tol=divergence_tol,
            )
            self.recovery.thresholds = _SimParams()
            self.recovery.thresholds.table = thresholds_table
            self.recovery.thresholds.summary = thresholds_summary

            gen_widths = self._category_widths(gen_thresholds)
            est_widths = self._category_widths(self.thresholds)
            widths_table, widths_summary = self._parameter_recovery_table(
                gen_widths, est_widths, se=self.cat_width_se, alpha=alpha,
                correction=correction, divergence_tol=divergence_tol,
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
            pseudo_estimate, self.persons, se=self.csem_vector, alpha=alpha,
            correction=correction, divergence_tol=divergence_tol,
        )
        self.recovery.persons = _SimParams()
        self.recovery.persons.table = persons_table
        self.recovery.persons.summary = persons_summary

        # Parameter-estimation residual (Elliott & Buttery, 2022): RMS
        # expected-score residual between (generating item location,
        # generating shared thresholds) and (estimated item location,
        # estimated shared thresholds), jointly -- at the true generating
        # theta throughout. Only meaningful when thresholds/category_widths
        # themselves were (gen_thresholds is not None) -- see
        # _parameter_estimation_residual.
        self.recovery.parameter_estimation_residual = None
        if gen_thresholds is not None:
            gen_items_ = self.generating.items
            self.recovery.parameter_estimation_residual = self._parameter_estimation_residual(
                self.generating.persons,
                lambda theta, item: self.exp_score(theta, gen_items_[item], gen_thresholds),
                lambda theta, item: self.exp_score(theta, est_items[item], self.thresholds),
            )

        if plot:
            self._recovery_plot(self.generating.items, est_items, "items", **plot_kwargs)
            if gen_thresholds is not None:
                self._recovery_plot(
                    gen_thresholds, self.thresholds, "thresholds", **plot_kwargs
                )
                self._recovery_plot(
                    gen_widths, est_widths, "category widths", **plot_kwargs
                )
            self._recovery_plot(pseudo_estimate, self.persons, "persons", **plot_kwargs)

        return None
