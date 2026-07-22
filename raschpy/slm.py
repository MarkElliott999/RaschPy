from math import log, sqrt
import warnings
from scipy.stats import chi2, norm, t as t_dist

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from matplotlib import pyplot as plt
from matplotlib import colors as colors
from matplotlib import cm as cmx
import seaborn as sns

from raschpy.base import Rasch


class SLM(Rasch):

    def __init__(
        self,
        responses,
        extreme_persons=True,
        no_of_classes=5,
        validate=True,
        exogenous=None,
    ):
        """
        Initialise a Simple Logistic Model (dichotomous Rasch model) object.

        Parameters
        ----------
        responses : pandas.DataFrame
            Response data with persons as rows and items as columns.
            Cell values should be 0 (incorrect) or 1 (correct); NaN for
            missing responses. The index is used as person identifiers and
            the columns as item identifiers.
        extreme_persons : bool, default True
            If True, removes only persons with entirely missing data.
            If False, additionally removes persons with all-zero or
            perfect scores (extreme scores), estimation for whom cannot
            be handled by maximum likelihood estimation without adjustment.
        no_of_classes : int, default 5
            Number of class intervals used in observed-data overlays on
            ICC, CRCs and TCC plots.
        validate : bool, default True
            If True, checks whether the item response network is fully
            connected (i.e. all items are linked via common persons).
            Issues a UserWarning if the data is split into disconnected
            sub-networks, which makes item locations incomparable
            across sub-groups.
        exogenous : pandas.DataFrame or None, default None
            Optional person-level covariates (e.g. Gender, Nationality
            or Age) for differential item functioning (DIF) analysis,
            indexed by person identifier. Values are kept as raw category
            labels. Persons in responses without a matching exogenous
            record (and vice versa) are allowed — such gaps are common
            when exogenous data is optional (e.g. for GDPR reasons) —
            and are reported via UserWarning plus the attributes
            exogenous_only_persons and no_exogenous_persons, rather than
            raising.

        Attributes set
        --------------
        responses : pandas.DataFrame
            Filtered response data (extreme/invalid persons removed).
        invalid_responses : pandas.DataFrame
            Rows removed from responses based on the extreme_persons rule.
        no_of_items : int
            Number of items in the filtered responses.
        no_of_persons : int
            Number of persons in the filtered responses.
        items : pandas.Index
            Item identifiers (column names of responses).
        persons : pandas.Index
            Person identifiers (index of responses).
        no_of_classes : int
            Number of class intervals (passed through for plot methods).
        max_score : int
            Always 1 for SLM (dichotomous model).
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

        super().__init__()

        # Sim-aware instantiation: store sim attributes in self.generating namespace
        from raschpy.simulation.slm_sim import SLM_Sim
        from raschpy.base import _SimParams

        if isinstance(responses, SLM_Sim):
            sim = responses
            self.generating = _SimParams()
            for attr, value in vars(sim).items():
                setattr(self.generating, attr, value)
            responses = sim.responses

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
            row_sums = valid.sum(axis=1)
            row_counts = valid.count(axis=1)
            extreme_mask = (row_sums == 0) | (row_sums == row_counts)
            self.extreme_persons = valid[extreme_mask]
            self.responses = valid[~extreme_mask]

        # Set foundational metadata attributes
        self.no_of_items = self.responses.shape[1]
        self.no_of_persons = self.responses.shape[0]

        self.item_names = self.responses.columns
        self.person_names = self.responses.index
        self.no_of_classes = no_of_classes
        self.max_score = 1

        # Optional person-level covariates for DIF (e.g. Gender, Nationality, Age)
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
                    f"⚠️  DATA INTEGRITY WARNING: {len(directionally_isolated)} item(s) have no "
                    f"zero-scored or no maximum-scored responses: {directionally_isolated}.\n"
                    f"These items pass the standard connectivity check (they have at least one "
                    f"empirical comparison in some direction) but have a structurally unresolvable "
                    f"zero in calibrate()'s directed pairwise matrix. This can silently produce NaN "
                    f"or overflow during calibration rather than a clear error. Consider dropping "
                    f"these items or gathering more responses before calibrating.",
                    category=UserWarning,
                    stacklevel=2,
                )

    def exp_score(self, person_location, item_location):
        """
        Compute the expected score (probability of correct response).

        Implements the SLM response function P(X=1) = 1 / (1 + exp(d - b)),
        where b is person location and d is item location. Fully vectorised:
        accepts scalars, 1-D arrays, or 2-D arrays in any combination that
        NumPy can broadcast.

        Parameters
        ----------
        person_location : float or array-like
            Person location(s) on the logit scale.
        item_location : float or array-like
            Item location(s) on the logit scale.

        Returns
        -------
        float or numpy.ndarray
            Expected score (probability of correct response), in [0, 1].
        """
        return 1.0 / (1.0 + np.exp(item_location - person_location))

    def cat_prob(self, person_location, item_location, category):
        """
        Compute the probability of a given response category (0 or 1).

        Returns P(X=category | person location, item location) using the SLM
        response function. For category=1 this is identical to exp_score;
        for category=0 it is 1 minus that probability.

        Parameters
        ----------
        person_location : float or array-like
            Person location estimate(s) on the logit scale.
        item_location : float or array-like
            Item location estimate(s) on the logit scale.
        category : int
            Response category: 0 (incorrect) or 1 (correct).

        Returns
        -------
        float or numpy.ndarray
            Probability of the specified response category, in [0, 1].
        """
        p = self.exp_score(person_location, item_location)
        # For category 1: returns p. For category 0: returns (1 - p).
        return p if category == 1 else (1.0 - p)

    def variance(self, person_location, item_location):
        """
        Compute the item response variance (Fisher information) for a given
        person location.

        For the SLM, variance equals P(X=1) * P(X=0) = p * (1 - p), which
        is simultaneously the Fisher information, the variance of the response
        distribution, and the first derivative of the expected score function
        with respect to person location.

        Parameters
        ----------
        person_location : float or array-like
            Person location estimate(s) on the logit scale.
        item_location : float or array-like
            Item location estimate(s) on the logit scale.

        Returns
        -------
        float or numpy.ndarray
            Response variance / Fisher information, in [0, 0.25].
        """
        expected = self.exp_score(person_location, item_location)
        return expected * (1.0 - expected)

    def kurtosis(self, person_location, item_location):
        """
        Compute the fourth central moment (kurtosis) of the response distribution.

        Used in the Wilson-Hilferty approximation for the standardised fit
        statistics (Infit Z, Outfit Z). For the SLM:
        kurtosis = p^4 * (1-p) + (1-p)^4 * p, where p = exp_score(person_location, item_location).

        Parameters
        ----------
        person_location : float or array-like
            Person location estimate(s) on the logit scale.
        item_location : float or array-like
            Item location estimate(s) on the logit scale.

        Returns
        -------
        float or numpy.ndarray
            Fourth central moment of the response distribution.
        """
        expected = self.exp_score(person_location, item_location)

        # Category 0 term: ((0 - expected)**4) * (1 - expected)
        term_1 = (expected**4) * (1.0 - expected)

        # Category 1 term: ((1 - expected)**4) * expected
        term_2 = ((1.0 - expected) ** 4) * expected

        return term_1 + term_2

    def _build_pairwise_matrix(self):
        """
        Raw (unsmoothed) directed pairwise comparison matrix used by
        calibrate() and check_data_connectivity(). Entry (i, j) counts
        persons who scored 1 on item i and 0 on item j.

        Returns
        -------
        matrix : numpy.ndarray, shape (no_of_items, no_of_items)
        row_items : numpy.ndarray
            Item name for each row/column (identity mapping for SLM).
        """
        df_array = np.array(self.responses, dtype=np.float64)
        is_one = ((df_array == 1) & (~np.isnan(df_array))).astype(np.float64)
        is_zero = ((df_array == 0) & (~np.isnan(df_array))).astype(np.float64)
        matrix = np.dot(is_one.T, is_zero)
        return matrix, np.array(self.item_names)

    def calibrate(
        self, constant=0.1, method="cos", matrix_power=3, log_lik_tol=0.000001
    ):
        """
        Estimate item locations using Choppin's pairwise matrix method.

        Constructs a pairwise comparison matrix from the response data and
        raises it to successive powers until all off-diagonal elements are
        non-zero (resolving structural zeroes that arise from items never
        administered together). A priority vector is then extracted from
        the resolved matrix to obtain item location estimates on the logit
        scale, zero-centred across all items.

        Issues a UserWarning if only one item is present (reduces to RSM)
        or if constant=0 and any item has all-maximum scores (estimation
        will fail without additive smoothing).

        Parameters
        ----------
        constant : float, default 0.1
            Additive smoothing constant added to structural zeroes remaining
            after matrix power resolution. Use 0 to disable smoothing, but
            note that this will cause estimation failure if any item has all
            maximum or all minimum scores.
        method : str, default 'cos'
            Method for extracting the priority vector from the pairwise
            matrix. 'cos' uses the cosine (geometric mean) method.
            See base.priority_vector() for full list of supported methods.
        matrix_power : int, default 3
            Initial power to which the pairwise matrix is raised before
            checking for structural zeroes. Higher values are more
            expensive but resolve zeroes faster on sparse data.
        log_lik_tol : float, default 0.000001
            Log-likelihood convergence tolerance passed to priority_vector()
            for methods that use iterative optimisation.

        Attributes set
        --------------
        diffs : pandas.Series
            Item location estimates indexed by item name, in logits,
            zero-centred across all items.
        null_persons : pandas.Index
            Persons dropped prior to calibration due to entirely missing
            response patterns.
        """

        if len(self.responses.columns) == 1:
            warnings.warn(
                "Only one item detected. This model with a single item reduces to RSM "
                "with raters as items. Consider reconfiguring and using RSM instead.",
                UserWarning,
                stacklevel=2,
            )

        if constant == 0:
            all_max_items = [
                item
                for item in self.responses.columns
                if self.responses[item].dropna().eq(self.responses[item].max()).all()
            ]

            if all_max_items:
                warnings.warn(
                    f"Items with all-maximum scores detected with constant=0: "
                    f"{list(all_max_items)}. Item estimation will fail. "
                    f"Either drop these items or use a non-zero constant.",
                    UserWarning,
                    stacklevel=2,
                )

        # 1. Clean up entirely empty rows (persons with zero data)
        self.null_persons = self.responses.index[self.responses.isnull().all(axis=1)]
        self.responses = self.responses.drop(self.null_persons)

        # 2. VECTORISED PAIRWISE COMPARISON MATRIX
        matrix, _ = self._build_pairwise_matrix()

        # 3. Compute matrix powers (Keep the diagonal as 0 so Choppin's math stays pure)
        # Sparse/disconnected resamples can blow this up to inf/nan before the zero-check
        # loop below terminates; the resulting isolated items are already surfaced via
        # check_data_connectivity's UserWarning, so the raw numpy RuntimeWarning is noise.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
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
        self.items = self.priority_vector(mat, method=method, log_lik_tol=log_lik_tol)

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
        constant=0.1,
        method="cos",
        matrix_power=3,
        log_lik_tol=0.000001,
        seed=None,
    ):
        """
        Anchor item location estimates onto externally-supplied values.

        Supports item banking: calibrates this dataset's own item
        locations as usual, then shifts the whole scale by a translation
        constant so that a subset of common ("anchor") items line up with
        externally-supplied reference locations (e.g. from a bank of
        previously-calibrated items). Since SLM item discrimination is
        fixed at 1, this is a simple mean-shift, not full linear equating.

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
        constant, method, matrix_power, log_lik_tol : floats
            Calibration/bootstrap kwargs, used only if calibrate or
            std_errors (selection_method='wald') is triggered.

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
        calibrate_anchor_runs : dict
            Every call's anchor_items/anchor_adj/anchor_summary/etc.,
            keyed by tuple(sorted(anchors.items())), so results from an
            earlier anchors call survive a later call with a different
            anchor set instead of being overwritten. E.g.
            slm.calibrate_anchor_runs[tuple(sorted(anchors_1.items()))].anchor_summary.
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
        )

    def std_errors(
        self,
        interval=None,
        no_of_samples=500,
        constant=0.1,
        method="cos",
        matrix_power=3,
        log_lik_tol=0.000001,
        seed=None,
    ):
        """
        Estimate bootstrap standard errors for item location estimates.

        Draws no_of_samples bootstrap resamples (with replacement) of the
        person-level response data, calibrates each resample, and computes
        the standard deviation of item location estimates across samples
        as the standard error. Optionally computes bootstrap confidence
        intervals.

        Parameters
        ----------
        interval : float or None, default None
            Confidence interval width, e.g. 0.95 for 95% CI. If None,
            only standard errors are computed. If provided, lower and
            upper percentile bounds are also stored.
        no_of_samples : int, default 500
            Number of bootstrap resamples. More samples give more stable
            SE estimates at the cost of computation time.
        constant : float, default 0.1
            Additive smoothing constant passed to calibrate() for each
            bootstrap resample.
        method : str, default 'cos'
            Priority vector extraction method passed to calibrate().
        matrix_power : int, default 3
            Matrix power passed to calibrate().
        log_lik_tol : float, default 0.000001
            Convergence tolerance passed to calibrate().
        seed : int or None, default None
            Seed for the bootstrap resampling RNG. Pass an int for fully
            reproducible standard errors; None (default) draws fresh entropy.

        Attributes set
        --------------
        item_se : pandas.Series
            Bootstrap standard error for each item location, indexed
            by item name.
        item_low : pandas.Series or None
            Lower percentile bound of the bootstrap CI for each item,
            or None if interval is None.
        item_high : pandas.Series or None
            Upper percentile bound of the bootstrap CI for each item,
            or None if interval is None.
        item_bootstrap : pandas.DataFrame
            Full matrix of bootstrap item location estimates, shape
            (no_of_samples, no_of_items), with items as columns and
            sample labels as index.
        bootstrap_sample_diffs : dict
            Dictionary of item location Series from each bootstrap resample,
            keyed by 'Sample_1', 'Sample_2', etc.
        """

        rng = np.random.default_rng(seed)
        samples = [
            SLM(self.responses.sample(frac=1, replace=True, random_state=rng), validate=False)
            for sample in range(no_of_samples)
        ]

        for sample in samples:
            sample.calibrate(
                constant=constant,
                method=method,
                matrix_power=matrix_power,
                log_lik_tol=log_lik_tol,
            )

        self.bootstrap_sample_diffs = {
            f"Sample_{i + 1}": sample.items for i, sample in enumerate(samples)
        }

        item_ests = np.array(
            [sample.items.loc[self.item_names].values for sample in samples]
        )

        self.item_se = {
            item: se for item, se in zip(self.item_names, np.nanstd(item_ests, axis=0))
        }
        self.item_se = pd.Series(self.item_se)

        if interval is not None:
            self.item_low = {
                item: low
                for item, low in zip(
                    self.item_names,
                    np.nanpercentile(item_ests, (1 - interval) * 50, axis=0),
                )
            }
            self.item_low = pd.Series(self.item_low)

            self.item_high = {
                item: high
                for item, high in zip(
                    self.item_names,
                    np.nanpercentile(item_ests, (1 + interval) * 50, axis=0),
                )
            }
            self.item_high = pd.Series(self.item_high)

        else:
            self.item_low = None
            self.item_high = None

        self.item_bootstrap = pd.DataFrame(item_ests)
        self.item_bootstrap.columns = self.responses.columns
        self.item_bootstrap.index = [f"Sample {i + 1}" for i in range(no_of_samples)]

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

        For each person, iteratively solves the likelihood equation
        sum(P_i) = score for the person location estimate, where P_i is the
        probability of a correct response on item i. Extreme scores
        (all-zero or perfect) are adjusted by ext_score_adjustment before
        estimation. Optionally applies Warm's (1989) bias correction.

        Parameters
        ----------
        persons : str or list
            Person identifier(s) to estimate person locations for. Pass 'all'
            to estimate for all persons in the responses.
        items : str or list or None, default None
            Item subset to use for estimation. None uses all items.
            Pass 'all' or a list of item names to restrict the item set.
        warm_corr : bool, default True
            If True, applies Warm's (1989) weighted maximum likelihood
            bias correction after Newton-Raphson convergence.
        tolerance : float, default 0.00001
            Convergence criterion: iteration stops when the maximum
            absolute change in person location estimates falls below this value.
        max_iters : int, default 100
            Maximum number of Newton-Raphson iterations. A UserWarning is
            raised if this limit is reached before convergence.
        ext_score_adjustment : float, default 0.5
            Amount added to (or subtracted from) extreme scores of 0 or
            maximum before estimation, to allow finite person location estimates.

        Returns
        -------
        pandas.Series
            Person location estimates indexed by person identifier, in logits.
            Returns numpy.nan for persons where estimation fails.
        """

        if isinstance(persons, str):
            if persons == "all":
                persons = self.person_names

            else:
                persons = [persons]

        if isinstance(items, str):
            if items == "all":
                items = self.item_names

            else:
                items = [items]

        if items is None:
            items = self.item_names
            item_locations = self.items
            person_data = self.responses.loc[persons]

        else:
            item_locations = self.items.loc[items]
            person_data = self.responses.loc[persons, items]

        if missing_as_incorrect:
            person_data = person_data.fillna(0)

        person_filter = (person_data + 1) / (person_data + 1)
        scores = person_data.sum(axis=1).astype(float)
        ext_scores = person_filter.sum(axis=1)

        scores[scores == 0] += ext_score_adjustment
        scores[scores == ext_scores] -= ext_score_adjustment

        item_location_df = pd.DataFrame(
            np.tile(item_locations.values[None, :], (len(persons), 1)),
            index=persons,
            columns=item_locations.index,
        )
        item_location_df *= person_filter

        try:
            estimates = (
                np.log(scores) - np.log(ext_scores - scores) + item_location_df.mean(axis=1)
            )
            changes = pd.Series({person: 1 for person in persons})
            iters = 0

            while (abs(changes).max() > tolerance) & (iters <= max_iters):
                exp_score_df = pd.DataFrame(
                    1
                    / (
                        1
                        + np.exp(
                            item_locations.values[None, :] - estimates.values[:, None]
                        )
                    ),
                    index=persons,
                    columns=item_locations.index,
                )

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
                estimates += self.warm(estimates, item_locations, person_filter)

            if iters >= max_iters:
                warnings.warn(
                    "Maximum iterations reached before convergence in person(). "
                    "Returned estimates may be inaccurate.",
                    UserWarning,
                    stacklevel=2,
                )

        except Exception:
            estimates = np.nan

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

        Convenience wrapper around person() that estimates person locations for every
        person in the responses and stores the result as self.persons.

        Parameters
        ----------
        items : str or list or None, default None
            Item subset to use. None uses all items. Pass 'all' or a
            list of item names to restrict the item set.
        warm_corr : bool, default True
            If True, applies Warm's (1989) bias correction.
        tolerance : float, default 0.00001
            Newton-Raphson convergence tolerance.
        max_iters : int, default 100
            Maximum Newton-Raphson iterations.
        ext_score_adjustment : float, default 0.5
            Adjustment applied to extreme scores before estimation.
        missing_as_incorrect : bool, default False
            If True, treats missing responses as score 0 rather than
            excluding them from the likelihood. Relevant for educational
            testing contexts where non-response implies incorrect.

        Attributes set
        --------------
        person_locations : pandas.Series
            Person location estimates for all persons, indexed by person identifier,
            in logits.
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
        Convert a raw score to a person location estimate via Newton-Raphson ML.

        Estimates the person location corresponding to a given integer raw score
        on a specified item set. Unlike person(), which operates on observed
        person response patterns, this method works from a scalar score
        and is used to draw score lines on TCC plots.

        Parameters
        ----------
        score : int or float
            Raw score to convert to a person location estimate. Extreme scores
            of 0 or maximum are adjusted by ext_score_adjustment.
        items : str or list or None, default None
            Item subset defining the score scale. None uses all items.
        warm_corr : bool, default True
            If True, applies Warm's (1989) bias correction.
        tolerance : float, default 0.00001
            Newton-Raphson convergence tolerance.
        max_iters : int, default 100
            Maximum Newton-Raphson iterations.
        ext_score_adjustment : float, default 0.5
            Adjustment applied to extreme scores (0 or maximum).

        Returns
        -------
        float
            Person location estimate in logits corresponding to the given score.
        """

        if items is None:
            items = self.item_names

        if isinstance(items, str):
            if items == "all":
                items = self.item_names

        item_locations = self.items.loc[items]

        person_filter = np.ones(len(items))
        max_score = len(item_locations)

        if score == 0:
            score = ext_score_adjustment

        elif score == max_score:
            score -= ext_score_adjustment

        estimate = log(score) - log(max_score - score) + item_locations.mean()
        change = 1
        iters = 0

        diffs_arr = item_locations.values

        while (abs(change) > tolerance) & (iters <= max_iters):

            p = 1.0 / (1.0 + np.exp(diffs_arr - estimate))
            result = p.sum()
            info = (p * (1.0 - p)).sum()

            change = max(-1, min(1, (result - score) / info))
            estimate -= change
            iters += 1

        if warm_corr:
            estimate += self.warm(estimate, item_locations, person_filter)

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

        Estimates the person location corresponding to every possible raw score on a
        given item set using vectorised Newton-Raphson, and stores the result
        as self.score_table. Useful for converting raw scores to person location
        estimates in batch without per-person response patterns.

        Parameters
        ----------
        items : str or list or None, default None
            Item subset to use. None uses all items. Pass 'all' or a
            list of item names.
        ext_scores : bool, default True
            If True, includes extreme scores (0 and maximum) in the table,
            adjusted by ext_score_adjustment. If False, only non-extreme
            scores are included.
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
        person_table : pandas.Series
            Person location estimate for each possible raw score, indexed by score.
        """

        if isinstance(items, str):
            if items == "all":
                items = self.item_names

            else:
                items = [items]

        if items is None:
            items = self.item_names

        no_of_items = len(items)
        item_locations = self.items.loc[items]

        if ext_scores:
            scores = np.arange(no_of_items + 1)

            used_scores = scores.astype(float)
            used_scores[0] += ext_score_adjustment
            used_scores[-1] -= ext_score_adjustment

        else:
            scores = np.arange(1, no_of_items)
            used_scores = scores.astype(float)

        estimates = {
            score: np.log(used_score)
            - np.log(no_of_items - used_score)
            + item_locations.mean()
            for score, used_score in zip(scores, used_scores)
        }
        estimates = pd.Series(estimates, index=scores)

        changes = pd.Series(1, index=scores)
        iters = 0

        while (abs(changes).max() > tolerance) & (iters <= max_iters):
            exp_score_df = pd.DataFrame(
                1
                / (
                    1 + np.exp(item_locations.values[None, :] - estimates.values[:, None])
                ),
                index=scores,
                columns=item_locations.index,
            )

            info_df = exp_score_df * (1 - exp_score_df)

            result_list = exp_score_df.sum(axis=1)
            info_list = info_df.sum(axis=1)

            changes = (result_list - used_scores) / info_list
            changes = changes.clip(-1, 1)
            estimates -= changes
            iters += 1

        if warm_corr:
            person_filter = pd.DataFrame(1, columns=items, index=scores)
            estimates += self.warm(estimates, item_locations, person_filter)

        self.score_table = estimates

    def warm(self, person_locations, item_locations, person_filter):
        """
        Apply Warm's (1989) weighted maximum likelihood bias correction.

        Computes the correction term J / (2 * I^2) as described in
        Warm (1989), where I is the total Fisher information and J is
        the sum of the third derivatives of the log-likelihood. Fully
        vectorised for simultaneous correction of multiple persons.
        Accepts either scalar or array inputs for persons.

        Parameters
        ----------
        person_locations : float or pandas.Series
            Current person location estimate(s). If a Series, index must match persons.
        item_locations : pandas.Series
            Item location estimates, indexed by item name.
        person_filter : numpy.ndarray or pandas.DataFrame
            Binary mask indicating which items each person responded to
            (1 = responded, NaN = missing). Shape must be compatible with
            broadcasting against person_locations and item_locations.

        Returns
        -------
        float or pandas.Series
            Warm bias correction term(s) to be added to the ML person location
            estimate(s). Same type and shape as the person_locations input.
        """

        if np.isscalar(person_locations):
            p = 1.0 / (1.0 + np.exp(item_locations.values - person_locations))
            info = p * (1.0 - p)
            i = (info * person_filter).sum()
            j = (info * (1 - 2 * p) * person_filter).sum()
            return j / (2 * i**2)

        exp_score_df = pd.DataFrame(
            1 / (1 + np.exp(item_locations.values[None, :] - person_locations.values[:, None])),
            index=person_locations.index,
            columns=item_locations.index,
        )

        info_df = exp_score_df * (1 - exp_score_df)

        j_df = info_df * (1 - 2 * exp_score_df)

        exp_score_df *= person_filter
        info_df *= person_filter
        j_df *= person_filter

        i = info_df.sum(axis=1)
        j = j_df.sum(axis=1)

        return j / (2 * i**2)

    def csem(self, persons=None, person_locations=None, items=None):
        """
        Compute the conditional standard error of measurement.

        Calculates CSEM = 1 / sqrt(I), where I is the total Fisher
        information per person given their item responses and person
        location estimate. A hypothetical location with no matching row in
        self.responses (e.g. a raw score used as a key into a custom
        person_locations lookup such as self.score_table) has no observed
        responses to consult, so all items in items are treated as answered.

        Parameters
        ----------
        persons : list, str, or None, default None
            Person identifiers. Overrides person_locations if provided.
        person_locations : pandas.Series, float, list, numpy.ndarray, or None, default None
            Person location estimates. If None, uses self.persons, calling
            self.person_estimates() automatically to generate it if not
            already present.
        items : list or None, default None
            Item subset to use. None uses all items.

        Returns
        -------
        pandas.Series
            Conditional standard error of measurement in logits, indexed by
            person (or by location label, for hypothetical locations).
        """
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

        if items is None:
            items = list(self.item_names)
        elif isinstance(items, str):
            items = [items]

        item_locations = self.items.loc[items]

        # Hypothetical locations (no matching row in self.responses) are
        # treated as fully answered; real persons are filtered by their
        # actual missing-response pattern.
        is_real_person = persons.isin(self.responses.index)
        person_filter = self.responses.reindex(persons)[items].notna().astype(float)
        person_filter.loc[~is_real_person] = 1.0

        diffs_arr = item_locations.values
        locs = person_locations.values
        p = 1.0 / (1.0 + np.exp(diffs_arr[None, :] - locs[:, None]))
        total_info = (p * (1.0 - p) * person_filter.values).sum(axis=1)

        return pd.Series(1.0 / np.sqrt(total_info), index=persons)

    def category_counts_df(self, persons=None, items=None, counts_name=None):
        """
        Build a response frequency table for one or more persons, across
        one or more items.

        Computes the number of 0 and 1 responses, total valid responses,
        and missing responses for each requested item, over the requested
        persons. Appends a 'Total' row summing across them.

        Note (breaking change from earlier versions): the valid-response-
        count column was renamed from 'Responses' to 'Total', matching
        PCM/RSM/MFRM's category_counts_df.

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
            DataFrame with items as rows and response categories (0, 1),
            'Total', and 'Missing' as columns. All values are
            integers. A 'Total' row is appended at the bottom.
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

        cat_counts_dict = {
            item: subset[item]
            .value_counts()
            .reindex([0, 1], fill_value=0)
            .astype(int)
            for item in items
        }
        category_counts_df = pd.DataFrame(cat_counts_dict).T

        category_counts_df["Total"] = subset.count()
        category_counts_df["Missing"] = (
            len(persons) - category_counts_df["Total"]
        )

        category_counts_df = category_counts_df.astype(int)

        category_counts_df.loc["Total"] = category_counts_df.sum()

        if counts_name is None:
            self.category_counts_table = category_counts_df
        else:
            if not hasattr(self, "counts"):
                from raschpy.base import _Namespace

                self.counts = _Namespace()
            self.counts[counts_name] = category_counts_df

        return category_counts_df

    def _log_likelihood(self, responses=None, persons=None):
        if responses is None:
            responses = self.responses
        if persons is None:
            persons = self.persons
        scores = responses.sum(axis=1)
        max_scores = responses.notna().sum(axis=1)
        non_extreme = responses.index[(scores > 0) & (scores < max_scores)]
        persons = persons.reindex(non_extreme).dropna()
        obs = responses.loc[persons.index].values
        p1 = 1 / (1 + np.exp(self.items.values[None, :] - persons.values[:, None]))
        p0 = 1 - p1
        valid = ~np.isnan(obs)
        prob_obs = np.where(obs == 1, p1, p0)
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
        constant=0.1,
        method="cos",
        matrix_power=3,
        log_lik_tol=0.000001,
        no_of_samples=500,
        interval=None,
        seed=None,
    ):
        """
        Compute all item, person, and test-level fit statistics.

        This is the central computation method. It auto-triggers calibrate(),
        std_errors(), and person_estimates() if they have not already been run.
        Computes expected scores, information, kurtosis, residuals, and
        standardised residuals for all person-item combinations, then derives
        Infit and Outfit mean-square and standardised (Z) statistics using
        the Wilson-Hilferty approximation. Also computes person separation,
        strata, and reliability if test_stats=True.

        Parameters
        ----------
        warm_corr : bool, default True
            If True, applies Warm's (1989) bias correction to person location
            estimates used in fit computation.
        se : bool, default True
            If True, computes bootstrap standard errors (required for
            test-level statistics). If False, test_stats is forced False.
        test_stats : bool, default True
            If True, computes test-level statistics (ISI, PSI, strata,
            reliability). Requires se=True.
        trim_cat_prob_dict : bool, default False
            If True, restricts cat_prob_dict to persons with non-extreme
            scores. Reduces memory usage on large datasets.
        tolerance : float, default 0.00001
            Newton-Raphson convergence tolerance for person location estimation.
        max_iters : int, default 100
            Maximum Newton-Raphson iterations for person location estimation.
        ext_score_adjustment : float, default 0.5
            Extreme score adjustment for person location estimation.
        constant : float, default 0.1
            Additive smoothing constant for calibration.
        method : str, default 'cos'
            Priority vector extraction method for calibration.
        matrix_power : int, default 3
            Matrix power for calibration.
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
            {0: DataFrame, 1: DataFrame} of category probabilities for
            all persons and items.
        exp_score_df : pandas.DataFrame
            Expected scores (non-extreme persons only), shape (persons, items).
        info_df : pandas.DataFrame
            Fisher information values, shape (persons, items).
        kurtosis_df : pandas.DataFrame
            Fourth central moments, shape (persons, items).
        residual_df : pandas.DataFrame
            Raw residuals (observed - expected), shape (persons, items).
        std_residual_df : pandas.DataFrame
            Standardised residuals (residual / sqrt(info)), shape (persons, items).
        item_infit_ms : pandas.Series
            Item infit mean-square statistics.
        item_outfit_ms : pandas.Series
            Item outfit mean-square statistics.
        item_infit_zstd : pandas.Series
            Item infit Z statistics (Wilson-Hilferty approximation).
        item_outfit_zstd : pandas.Series
            Item outfit Z statistics.
        response_counts : pandas.Series
            Number of valid responses per item.
        item_facilities : pandas.Series
            Mean response (proportion correct) per item.
        point_measure : pandas.Series
            Observed point-measure correlations per item.
        exp_point_measure : pandas.Series
            Expected point-measure correlations per item.
        discrimination : pandas.Series
            Item discrimination indices.
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
        isi : float
            Item separation index (if test_stats=True).
        item_strata : float
            Number of statistically distinct item strata (if test_stats=True).
        item_reliability : float
            Item reliability coefficient (if test_stats=True).
        psi : float
            Person separation index (if test_stats=True).
        person_strata : float
            Number of statistically distinct person strata (if test_stats=True).
        person_reliability : float
            Person reliability coefficient (if test_stats=True).
        item_residual_corr : pandas.Series
            Correlation of standardised residuals with item locations.
        person_residual_corr : pandas.Series
            Correlation of standardised residuals with person locations.
        """

        if not hasattr(self, "items"):
            self.calibrate(
                constant=constant,
                method=method,
                matrix_power=matrix_power,
                log_lik_tol=log_lik_tol,
            )

        if se:
            if not hasattr(self, "item_se"):
                self.std_errors(
                    interval=interval,
                    no_of_samples=no_of_samples,
                    constant=constant,
                    method=method,
                    matrix_power=matrix_power,
                    log_lik_tol=log_lik_tol,
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

        """
        Create matrices of expected scores, variances, kurtosis, residuals etc. to generate fit statistics
        """

        # BUG FIX: (df == df) is the old NaN-detection idiom; .notna() is cleaner
        # and consistent with PCM/RSM.
        item_count = self.responses.notna().sum(axis=0)
        person_count = self.responses.notna().sum(axis=1)

        df = self.responses.copy()
        scores = df.sum(axis=1)
        max_scores = df.notna().sum(axis=1)

        df = df[(scores > 0) & (scores < max_scores)]
        missing_mask = (df + 1) / (df + 1)

        p = pd.DataFrame(
            1 / (1 + np.exp(self.items.values[None, :] - self.persons.values[:, None])),
            index=self.person_names,
            columns=self.item_names,
        )
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
        self.kurtosis_df = p0 * (p1**4) + p1 * (p0**4)
        self.kurtosis_df *= missing_mask

        self.log_likelihood = self._log_likelihood()

        n_persons = int(((scores > 0) & (scores < max_scores)).sum())
        k = self.no_of_items - 1
        self.aic = 2 * k - 2 * self.log_likelihood
        self.bic = k * np.log(n_persons) - 2 * self.log_likelihood

        self.residual_df = self.responses - self.exp_score_df
        self.std_residual_df = self.residual_df / (self.info_df**0.5)

        self.exp_score_df = self.exp_score_df[(scores > 0) & (scores < max_scores)]
        self.info_df = self.info_df[(scores > 0) & (scores < max_scores)]
        self.kurtosis_df = self.kurtosis_df[(scores > 0) & (scores < max_scores)]
        self.residual_df = self.residual_df[(scores > 0) & (scores < max_scores)]
        self.std_residual_df = self.std_residual_df[
            (scores > 0) & (scores < max_scores)
        ]

        """
        Item fit statistics
        """

        self.item_outfit_ms = (self.std_residual_df**2).mean()
        self.item_infit_ms = (self.residual_df**2).sum() / self.info_df.sum()

        item_outfit_q = (
            (self.kurtosis_df / (self.info_df**2))
            / (item_count.loc[self.kurtosis_df.columns] ** 2)
        ).sum() - (1 / item_count.loc[self.kurtosis_df.columns])
        item_outfit_q = item_outfit_q**0.5
        self.item_outfit_zstd = ((self.item_outfit_ms ** (1 / 3)) - 1) * (
            3 / item_outfit_q
        ) + (item_outfit_q / 3)

        item_infit_q = (self.kurtosis_df - self.info_df**2).sum() / (
            self.info_df.sum() ** 2
        )
        item_infit_q = item_infit_q**0.5
        self.item_infit_zstd = ((self.item_infit_ms ** (1 / 3)) - 1) * (
            3 / item_infit_q
        ) + (item_infit_q / 3)

        self.response_counts = self.responses.count(axis=0)
        self.item_facilities = self.responses.mean(axis=0)

        (self.point_measure, self.exp_point_measure) = self.pt_meas(
            self.persons, self.exp_score_df, self.info_df
        )

        """
        Person fit statistics
        """

        self.csem_vector = 1 / (self.info_df.sum(axis=1) ** 0.5)
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
        base_df = base_df.div(person_count.loc[base_df.index] ** 2, axis=0)

        person_outfit_q = base_df.sum(axis=1) - (1 / person_count.loc[base_df.index])
        person_outfit_q = person_outfit_q**0.5
        self.person_outfit_zstd = ((self.person_outfit_ms ** (1 / 3)) - 1) * (
            3 / person_outfit_q
        ) + (person_outfit_q / 3)
        self.person_outfit_zstd.name = "Outfit Z"

        person_infit_q = (self.kurtosis_df - self.info_df**2).sum(axis=1) / (
            self.info_df.sum(axis=1) ** 2
        )
        person_infit_q = person_infit_q**0.5
        self.person_infit_zstd = ((self.person_infit_ms ** (1 / 3)) - 1) * (
            3 / person_infit_q
        ) + (person_infit_q / 3)
        self.person_infit_zstd.name = "Infit Z"

        differences = pd.DataFrame(
            self.persons.values[:, None] - self.items.values[None, :],
            index=self.person_names,
            columns=self.item_names,
        )
        differences = differences.loc[self.residual_df.index]
        num = (differences * self.residual_df).sum(axis=0)
        den = (self.info_df * (differences**2)).sum(axis=0)
        self.discrimination = 1 + num / den

        """
        Test-level fit statistics
        """

        if test_stats:
            item_ests = self.item_bootstrap.values
            isi_samples = (
                np.var(item_ests, axis=1, ddof=1)
                / np.var(
                    item_ests - item_ests.mean(axis=1, keepdims=True), axis=0
                ).mean()
            )
            self.isi = np.sqrt(np.mean(isi_samples) - 1)

            self.item_strata = (4 * self.isi + 1) / 3
            self.item_reliability = self.isi**2 / (1 + self.isi**2)

            psi_var = max(0, np.var(self.persons) - (self.rsem_vector**2).mean())
            self.psi = (psi_var**0.5) / ((self.rsem_vector**2).mean() ** 0.5)

            self.person_strata = (4 * self.psi + 1) / 3
            self.person_reliability = (self.psi**2) / (1 + (self.psi**2))

        self.item_residual_corr = self.std_residual_df.corrwith(self.items, axis=1)
        self.person_residual_corr = self.std_residual_df.corrwith(
            self.persons.loc[self.std_residual_df.index], axis=0
        )

    def andersen_lr_test(
        self,
        split_by="person_location",
        covariate=None,
        warm_corr=True,
        tolerance=0.00001,
        max_iters=100,
        ext_score_adjustment=0.5,
        constant=0.1,
        method="cos",
        matrix_power=3,
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
            Degrees of freedom (no_of_items - 1).
        andersen_p : float
            p-value from chi-squared distribution.
        andersen_groups : dict
            {group_name: SLM} — fitted group models for inspection. Group
            names are 'low'/'high' for split_by='person_location'/'score', or the
            two observed covariate values for split_by='exogenous'.
        """
        from raschpy.slm import SLM

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

        if not hasattr(self, "items"):
            self.calibrate(
                constant=constant,
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

        # Non-extreme persons
        scores = self.responses.sum(axis=1)
        max_scores = self.responses.notna().sum(axis=1)
        non_extreme = self.responses.index[(scores > 0) & (scores < max_scores)]

        group_idx = self._resolve_andersen_groups(
            split_by, covariate, non_extreme, self.persons, scores
        )

        # Full-model LL restricted to persons in either group
        combined_idx = group_idx[list(group_idx)[0]].append(group_idx[list(group_idx)[1]])
        # Re-estimate full model on combined subset so the LR comparison is fair
        m_full = SLM(self.responses.loc[combined_idx])
        m_full.calibrate(
            constant=constant,
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
            m = SLM(self.responses.loc[idx])
            m.calibrate(
                constant=constant,
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
        df = self.no_of_items - 1

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
        constant=0.1,
        method="cos",
        matrix_power=3,
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

        For each focal group: both groups are calibrated independently
        (each self-centred), then purified onto a common scale before
        testing — so that genuine DIF items cannot contaminate the scale
        used to test for DIF in the first place — via either
        _wald_anchor_selection() (default) or _robust_anchor_selection().
        Item-level DIF is then tested on the purified scale for every item
        (not only the ones used to define the scale), via a per-item Wald
        test (test='wald', default), a per-item likelihood-ratio test
        (test='lr'), or both (test='both'). An omnibus LR test (Andersen-
        style: reference vs focal, all items jointly) runs by default
        (omnibus=True) alongside whichever per-item test(s) are chosen,
        answering "is there DIF at all" before the per-item table answers
        "in which items".

        The Wald test uses bootstrap standard errors from std_errors(),
        which are computed for both groups regardless of selection_method
        (needed for the per-item Wald test either way — so
        selection_method='wald' costs nothing extra here, unlike
        calibrate_anchor where it triggers an otherwise-unneeded
        bootstrap). The LR test (per-item and omnibus) instead needs
        person locations, which this method does NOT otherwise estimate —
        so test='lr'/'both' or omnibus=True adds person_estimates() calls
        that test='wald' with omnibus=False skips entirely.

        Per-item LR mechanics: H1 (every item free per group) log-
        likelihood is just ref_model._log_likelihood() +
        focal_model._log_likelihood() — each group's own natively-
        calibrated fit, computed once and reused for every item's test
        (translating one group's scale by a constant doesn't change its
        own internal fit, so no combined-dataset object is needed for
        this). For H0 on item i: replace item i's location in each
        group with a single value pooled across both groups (precision-
        weighted, on the reference/common scale, converted back to each
        group's own scale for the focal side), re-estimate that group's
        person locations under the modified item set, and recompute its log-
        likelihood. LR_i = 2*(LL_H1 - LL_H0_i), df=1. This is quite a bit
        more expensive than the Wald test (two person location re-estimations per
        item per focal-group comparison, versus none for Wald), though
        each individual re-estimation is typically fast.

        Parameters
        ----------
        covariate : str
            Column name in self.exogenous to group persons by.
        reference : str or None, default None
            Value of `covariate` to use as the reference group. If None,
            the largest group is used.
        selection_method : {'wald', 'robust_z', 'none'}, default 'wald'
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
            Per-item DIF test(s) to compute. 'wald' matches prior
            behaviour exactly (no added cost). 'lr'/'both' additionally
            compute the per-item likelihood-ratio test described above
            (extra cost: two person location re-estimations per item per focal
            group). The existing 'Flagged' column always reflects the
            Wald test (backward compatible); a separate 'Flagged_LR'
            column is added when test is 'lr' or 'both'.
        omnibus : bool, default True
            If True, also runs an Andersen-style omnibus LR test
            (reference vs. focal, every item jointly) per focal group,
            stored in dif_omnibus_table. Cheap relative to the per-item LR
            test — one extra combined-group model fit per focal group, not
            per item. The H1 side (ll_ref + ll_focal) plugs in person location
            estimates from the pooled reference+focal model rather than
            each group's own separately-fit person locations — otherwise the two
            sides of the comparison re-optimise nuisance person location parameters
            independently, which inflates the LR statistic beyond what
            df=k-1 accounts for (confirmed by null-DIF simulation: ~1.3-1.7x
            nominal Type I error with own-group person locations, ~nominal with
            pooled person locations).
        welch : bool, default False
            If True, the Wald test (test='wald' or 'both') becomes a
            Welch's t-test: the statistic is unchanged (diff / combined
            SE) but its p-value is computed against a t-distribution with
            Welch-Satterthwaite degrees of freedom (using each group's own
            person count) instead of a normal distribution — more
            conservative than a z-test when group sizes are small,
            unequal, or the two SEs differ substantially. Adds a 'df'
            column to dif_table. Does not affect the LR test or omnibus
            (already exact chi-squared tests, not z/t-based) or
            _wald_anchor_selection (a one-sample comparison against a
            fixed anchor value, not the two-independent-samples case
            Welch's t-test addresses).
        size_adjust : bool, default False
            If True, rescales each group's item SE to what it would be at
            a standard reference sample size (Tristán, 2006, Rasch
            Measurement Transactions 20:3 — an adjustment for the opposite
            problem from welch's: at very large N, SE shrinks enough that
            trivial differences become "significant", swamping the
            logit-magnitude thresholds' intent). Each group's own SE is
            rescaled independently by sqrt(actual_n / reference_n) using
            that group's own person count, then combined as usual — so
            unequal reference/focal group sizes are put on equal footing
            too, not just corrected for extremity. Applied to the Wald/
            Welch test, the category tests, and (if welch=True) the
            Welch-Satterthwaite df, which then also uses reference_n
            rather than each group's actual n, for internal consistency —
            everything behaves as if the data had been collected at
            reference_n. Does not affect the LR test's own precision-
            weighted pooling (a nuisance-parameter estimation detail, not
            the significance-test artefact this targets) or the omnibus
            test (an exact chi-squared test, not SE-based).
        reference_n : int, default 100
            Reference sample size for size_adjust=True. 100 is Tristán's own
            default recommendation; ~60 is suggested there for closer
            alignment with the ETS Category B boundary specifically
            (100*(0.43/0.55)**2). Unused unless size_adjust=True.
        correction : 'bh', 'bonferroni', or None, default 'bh'
            Multiple-comparison correction applied across items within
            each focal-group comparison (DIF flagging, not purification) —
            applied separately to the Wald and per-item LR p-values if
            both are computed. Not applied to the omnibus test (one test
            per focal group, not a family of item-level comparisons).
        alpha : float, default 0.05
            p-value threshold for DIF flagging (Wald, per-item LR, and
            omnibus alike). Compared against the corrected p-value where
            correction applies.
        logit_threshold : float, default 0.43
            Absolute logit-difference threshold for flagging (ETS-style
            convention). An item is flagged (Flagged or Flagged_LR) only if
            both this and the relevant p-value threshold are met — the
            LR-based flag reuses the same purified item-location Difference
            estimate as the Wald flag as its effect-size gate (the LR
            statistic itself has no natural logit-magnitude analogue, but
            the underlying quantity it's testing does).
        category : bool, default False
            If True, adds an ETS-style 'Category' column to dif_table:
            'A' (negligible), 'B+'/'B-' (slight to moderate), or 'C+'/'C-'
            (moderate to large), following Zwick, Thayer & Lewis (1999).
            Sign: '+' means Difference > 0 (item harder for focal, i.e.
            DIF against reference by the ETS convention as confirmed for
            this package); '-' means DIF against focal. Uses two tests,
            not one: 'B' requires |Difference| >= category_thresholds[0]
            AND prob(DIF=0) < category_alpha (this is just the existing
            Wald/Welch p-value, reused, not recomputed); 'C' requires
            |Difference| >= category_thresholds[1] AND a *different*,
            one-sided test — prob(|DIF| <= category_thresholds[0]) <
            category_alpha, i.e. whether |Difference| is significantly
            *above* the B/C boundary itself, not just significantly
            nonzero. Uses the same reference distribution as welch (t
            with Satterthwaite df, or normal). Independent of test=/
            logit_threshold=/Flagged — this reproduces the ETS scheme
            exactly, not a repackaging of the existing Wald flag. Scoped
            to dif_table only (item-location DIF) — the 0.43/0.64 logit
            defaults are specifically calibrated for item-location DIF
            in the literature, not threshold/step DIF.
        category_thresholds : (float, float), default (0.43, 0.64)
            (B boundary, C boundary) in logits, per Zwick et al. (1999).
            Override with your own values if you're working from a
            different source or convention.
        category_alpha : float, default 0.05
            Significance level for both of the category tests above — the
            ETS scheme's own literature-standard .05, kept independent of
            `alpha` (which governs Flagged/Flagged_LR) so overriding one
            doesn't silently change the other.
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
            at once isn't the right default.
        plot_kwargs : dict or None, default None
            Extra keyword arguments forwarded to plot_anchor_selection()
            for every focal group when plot=True (e.g. filename, xmin/
            xmax, title).
        warm_corr, tolerance, max_iters, ext_score_adjustment : floats
            Person estimation kwargs, passed through to group models. Only
            actually used when test in ('lr', 'both') or omnibus=True —
            with test='wald' and omnibus=False, no person locations are estimated
            at all and these are unused, but kept for signature
            consistency with other group-fitting methods such as
            andersen_lr_test.
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
            style joint test of every common item at once. None if
            omnibus=False.
        dif_reference : the reference group value.
        dif_covariate : the covariate column name used.
        dif_reference_model : SLM
            Fitted model for the reference group.
        dif_focal_models : dict {focal_value: SLM}
            Fitted models for each focal group.
        dif_tc : dict {focal_value: float}
            Translation constant applied to each focal group.
        dif_anchor_selection : dict {focal_value: pandas.DataFrame or None}
            Per-item selection diagnostics for each focal group (None if
            selection_method='none').
        dif_plots : dict {focal_value: matplotlib.figure.Figure or None}
            plot_anchor_selection() figure for each focal group, if
            plot=True and a selection table exists for that group; None
            otherwise.
        dif_group_sizes : dict {group_value: int}
            Non-extreme, non-missing-covariate N used for each group.
        """
        from raschpy.slm import SLM

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
        max_scores = self.responses.notna().sum(axis=1)
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
        ref_model = SLM(self.responses.loc[ref_idx], validate=False)
        ref_model.calibrate(
            constant=constant, method=method, matrix_power=matrix_power,
            log_lik_tol=log_lik_tol,
        )
        ref_model.std_errors(
            no_of_samples=no_of_samples, constant=constant, method=method,
            matrix_power=matrix_power, log_lik_tol=log_lik_tol, seed=seed,
        )
        if needs_ll:
            ref_model.person_estimates(**pe_kw)
            ll_ref = ref_model._log_likelihood()

        all_rows = []
        focal_models = {}
        tc_dict = {}
        anchor_selection_dict = {}
        plot_dict = {}
        omnibus_rows = {}

        for focal in focal_levels:
            focal_idx = cov_values.index[cov_values == focal]
            focal_model = SLM(self.responses.loc[focal_idx], validate=False)
            focal_model.calibrate(
                constant=constant, method=method, matrix_power=matrix_power,
                log_lik_tol=log_lik_tol,
            )
            focal_model.std_errors(
                no_of_samples=no_of_samples, constant=constant, method=method,
                matrix_power=matrix_power, log_lik_tol=log_lik_tol, seed=seed,
            )
            if needs_ll:
                focal_model.person_estimates(**pe_kw)
                ll_focal = focal_model._log_likelihood()

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
                welch_df = pd.Series(df_vals, index=common_items)
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
                        t_dist.cdf(-boundary_stat, welch_df.values), index=common_items
                    )
                else:
                    p_boundary = pd.Series(norm.cdf(-boundary_stat), index=common_items)

                category_col = []
                for item in common_items:
                    if abs_diff[item] >= c_thr and p_boundary[item] < category_alpha:
                        base = "C"
                    elif abs_diff[item] >= b_thr and p[item] < category_alpha:
                        base = "B"
                    else:
                        base = "A"
                    category_col.append(
                        base if base == "A" else base + ("+" if diff[item] > 0 else "-")
                    )
                category_col = pd.Series(category_col, index=common_items)

            if omnibus:
                combined_idx = ref_idx.append(focal_idx)
                m_full = SLM(self.responses.loc[combined_idx], validate=False)
                m_full.calibrate(
                    constant=constant, method=method, matrix_power=matrix_power,
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
                ref_scratch = SLM(ref_model.responses, validate=False)
                focal_scratch = SLM(focal_model.responses, validate=False)
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

            table = pd.DataFrame(
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
            table.index.name = "Item"
            if welch:
                table["df"] = welch_df
            if category:
                table["Category"] = category_col
            if test in ("lr", "both"):
                table = table.join(lr_table)

            all_rows.append(table)
            focal_models[focal] = focal_model
            tc_dict[focal] = tc

            if plot and anchor_selection_dict[focal] is not None:
                plot_dict[focal] = self.plot_anchor_selection(
                    anchor_selection_dict[focal], **(plot_kwargs or {})
                )
            else:
                plot_dict[focal] = None

        self.dif_table = pd.concat(all_rows)
        self.dif_omnibus_table = (
            pd.DataFrame(omnibus_rows).T if omnibus else None
        )
        self.dif_reference = reference
        self.dif_covariate = covariate
        self.dif_reference_model = ref_model
        self.dif_focal_models = focal_models
        self.dif_tc = tc_dict
        self.dif_anchor_selection = anchor_selection_dict
        self.dif_plots = plot_dict
        self.dif_group_sizes = group_sizes

    def res_corr_analysis(
        self,
        warm_corr=True,
        tolerance=0.00001,
        max_iters=100,
        ext_score_adjustment=0.5,
        constant=0.1,
        method="cos",
        matrix_power=3,
        log_lik_tol=0.000001,
        se=True,
    ):
        """
        Analyse standardised residual correlations for local item dependence.

        Computes the inter-item correlation matrix of standardised residuals
        and performs a Principal Component Analysis (PCA) on it to detect
        violations of local item independence and unidimensionality. A large
        first eigenvalue (conventionally > 2.0) suggests a second dimension
        in the data. Auto-triggers fit_statistics() if not yet run.

        Parameters
        ----------
        warm_corr : bool, default True
            Warm bias correction for person location estimates used in residuals.
        tolerance : float, default 0.00001
            Newton-Raphson convergence tolerance.
        max_iters : int, default 100
            Maximum Newton-Raphson iterations.
        ext_score_adjustment : float, default 0.5
            Extreme score adjustment for person location estimation.
        constant : float, default 0.1
            Additive smoothing constant for calibration.
        method : str, default 'cos'
            Priority vector extraction method for calibration.
        matrix_power : int, default 3
            Matrix power for calibration.
        log_lik_tol : float, default 0.000001
            Log-likelihood tolerance for calibration.
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
            Item-by-item correlation matrix of standardised residuals,
            shape (no_of_items, no_of_items).
        eigenvectors : pandas.DataFrame or None
            PCA eigenvectors (principal component loadings matrix),
            shape (no_of_items - 1, no_of_items), index labelled
            'PC 1' etc., columns labelled 'Eigenvector 1' etc.
            None if PCA fails.
        eigenvalues : pandas.DataFrame or None
            Eigenvalues for each principal component,
            shape (no_of_items - 1, 1), index labelled 'PC 1' etc.
            None if PCA fails.
        variance_explained : pandas.DataFrame or None
            Proportion of variance explained by each principal component,
            shape (no_of_items - 1, 1). None if PCA fails.
        loadings : pandas.DataFrame or None
            PCA loadings (eigenvectors scaled by sqrt(eigenvalue)),
            shape (no_of_items, no_of_items - 1), items as rows, PCs as columns.
            None if PCA fails.
        pca_fail : bool
            Set to True (only) if PCA raises an exception.
        """

        if not hasattr(self, "std_residual_df"):
            self.fit_statistics(
                se=se,
                warm_corr=warm_corr,
                tolerance=tolerance,
                max_iters=max_iters,
                ext_score_adjustment=ext_score_adjustment,
                constant=constant,
                method=method,
                matrix_power=matrix_power,
                log_lik_tol=log_lik_tol,
            )

        self.residual_correlations = self.std_residual_df.corr(numeric_only=False)

        pca = PCA()
        try:
            pca.fit(self.std_residual_df.corr())

            n = (
                self.no_of_items - 1
            )  # rank of correlation matrix is n-1; drop zero eigenvalue
            pc_labels = [f"PC {pc + 1}" for pc in range(n)]
            item_labels = list(self.responses.columns)

            self.eigenvectors = pd.DataFrame(pca.components_[:n, :])
            self.eigenvectors.index = pc_labels
            self.eigenvectors.columns = [
                f"Eigenvector {pc + 1}" for pc in range(self.no_of_items)
            ]

            self.eigenvalues = pd.DataFrame(pca.explained_variance_[:n])
            self.eigenvalues.index = pc_labels
            self.eigenvalues.columns = ["Eigenvalue"]

            self.variance_explained = pd.DataFrame(pca.explained_variance_ratio_[:n])
            self.variance_explained.index = pc_labels
            self.variance_explained.columns = ["Variance explained"]

            self.loadings = self.eigenvectors.T * (pca.explained_variance_[:n] ** 0.5)
            self.loadings = pd.DataFrame(self.loadings)
            self.loadings.index = item_labels
            self.loadings.columns = pc_labels

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

    def item_stats_df(
        self,
        full=False,
        zstd=False,
        disc=False,
        point_measure_corr=False,
        dp=3,
        se=True,
        warm_corr=True,
        tolerance=0.00001,
        max_iters=100,
        ext_score_adjustment=0.5,
        method="cos",
        constant=0.1,
        matrix_power=3,
        log_lik_tol=0.000001,
        no_of_samples=500,
        interval=None,
        seed=None,
    ):
        """
        Build and store the item statistics summary table.

        Auto-triggers std_errors() and fit_statistics() if not yet run.
        Always includes item location estimates, response counts,
        facilities, and Infit/Outfit MS. Additional columns (SE, Z
        statistics, discrimination, point-measure correlations, CI bounds)
        are included based on flags or when full=True.

        Parameters
        ----------
        full : bool, default False
            If True, sets zstd=True, disc=True, point_measure_corr=True,
            and interval=0.95. Overrides individual flags.
        zstd : bool, default False
            If True, includes Infit Z and Outfit Z columns.
        disc : bool, default False
            If True, includes item discrimination (Discrim) column.
        point_measure_corr : bool, default False
            If True, includes observed and expected point-measure
            correlation columns (PM corr, Exp PM corr).
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
        method : str, default 'cos'
            Priority vector extraction method for calibration.
        constant : float, default 0.1
            Additive smoothing constant for calibration.
        matrix_power : int, default 3
            Matrix power for calibration.
        log_lik_tol : float, default 0.000001
            Log-likelihood tolerance for calibration.
        no_of_samples : int, default 500
            Bootstrap samples for SE estimation. Unused if se=False.
        interval : float or None, default None
            Confidence interval width for bootstrap CIs (e.g. 0.95).
            If provided, lower and upper percentile columns are included.
            Ignored if se=False.
        seed : int or None, default None
            Seed passed through to the internal std_errors()/fit_statistics()
            calls (only used if not already computed). None draws fresh
            entropy each call.

        Attributes set
        --------------
        item_stats : pandas.DataFrame
            Item statistics table with items as rows. Always contains
            'Estimate', 'Count', 'Facility', 'Infit MS', 'Outfit MS'.
            Optional columns: 'SE' and CI bounds (if se=True), 'Infit Z',
            'Outfit Z', 'Discrim', 'PM corr', 'Exp PM corr'.
        """

        if full:
            zstd = True
            disc = True
            point_measure_corr = True

            if interval is None:
                interval = 0.95

        if not se:
            interval = None

        if se and (
            not hasattr(self, "item_se")
            or (interval is not None and not hasattr(self, "item_low"))
        ):
            self.std_errors(
                interval=interval,
                no_of_samples=no_of_samples,
                constant=constant,
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

        self.item_stats = pd.DataFrame()

        self.item_stats["Estimate"] = self.items.astype(float).round(dp)

        if se:
            self.item_stats["SE"] = self.item_se.astype(float).round(dp)

            if interval is not None:
                self.item_stats[f"{round((1 - interval) * 50, 1)}%"] = self.item_low.astype(
                    float
                ).round(dp)
                self.item_stats[f"{round((1 + interval) * 50, 1)}%"] = (
                    self.item_high.astype(float).round(dp)
                )

        self.item_stats["Count"] = self.response_counts.astype(int)
        self.item_stats["Facility"] = self.item_facilities.astype(float).round(dp)

        self.item_stats["Infit MS"] = self.item_infit_ms.astype(float).round(dp)
        if zstd:
            self.item_stats["Infit Z"] = self.item_infit_zstd.astype(float).round(dp)

        self.item_stats["Outfit MS"] = self.item_outfit_ms.astype(float).round(dp)
        if zstd:
            self.item_stats["Outfit Z"] = self.item_outfit_zstd.astype(float).round(dp)

        if disc:
            self.item_stats["Discrim"] = self.discrimination.astype(float).round(dp)

        if point_measure_corr:
            self.item_stats["PM corr"] = self.point_measure.astype(float).round(dp)
            self.item_stats["Exp PM corr"] = self.exp_point_measure.astype(float).round(
                dp
            )

        self.item_stats.index = self.responses.columns

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
        method="cos",
        constant=0.1,
    ):
        """
        Build and store the person statistics summary table.

        Auto-triggers fit_statistics() if not yet run. Produces one row
        per person with person location estimate, CSEM, raw score, maximum possible
        score, proportion correct, and Infit/Outfit MS and Z statistics.
        Persons with extreme scores are included with NaN fit statistics.

        Parameters
        ----------
        full : bool, default False
            If True, sets rsem=True. Overrides the rsem flag.
        rsem : bool, default False
            If True, includes the Residual SEM (RSEM) column.
        dp : int, default 3
            Number of decimal places for rounding numeric output.
        se : bool, default True
            Passed through to the internal fit_statistics() call (only
            used if not already computed). If False, skips the bootstrap
            entirely — this table's own columns (CSEM, RSEM, Infit/Outfit)
            do not depend on it, so se=False is purely a speed-up (e.g.
            for repeated simulation runs) with no effect on the output.
        warm_corr : bool, default True
            Warm bias correction for person location estimates.
        tolerance : float, default 0.00001
            Newton-Raphson convergence tolerance.
        max_iters : int, default 100
            Maximum Newton-Raphson iterations.
        ext_score_adjustment : float, default 0.5
            Extreme score adjustment for person location estimation.
        method : str, default 'cos'
            Priority vector extraction method for calibration.
        constant : float, default 0.1
            Additive smoothing constant for calibration.

        Attributes set
        --------------
        person_stats : pandas.DataFrame
            Person statistics table with persons as rows. Always contains
            'Estimate', 'CSEM', 'Score', 'Max score', 'p', 'Infit MS',
            'Infit Z', 'Outfit MS', 'Outfit Z'. Optional: 'RSEM'.
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

        person_stats_df = pd.DataFrame()
        person_stats_df.index = self.responses.index

        person_stats_df["Estimate"] = self.persons.round(dp)

        person_stats_df["CSEM"] = self.csem_vector.round(dp)
        if rsem:
            person_stats_df["RSEM"] = self.rsem_vector.round(dp)

        person_stats_df["Score"] = self.responses.sum(axis=1).astype(int)
        person_stats_df["Max score"] = self.responses.count(axis=1).astype(int)
        person_stats_df["p"] = self.responses.mean(axis=1).round(dp)

        # BUG FIX: .update(dict) ignores index alignment — extreme-score persons
        # (whose fit stats are NaN) would silently overwrite valid values when
        # the fit Series has a different index order. Use .loc[] instead.
        for col, src in [
            ("Infit MS", self.person_infit_ms),
            ("Infit Z", self.person_infit_zstd),
            ("Outfit MS", self.person_outfit_ms),
            ("Outfit Z", self.person_outfit_zstd),
        ]:
            person_stats_df[col] = np.nan
            person_stats_df.loc[src.index, col] = src.round(dp).values

        self.person_stats = person_stats_df

    def test_stats_df(
        self,
        dp=3,
        warm_corr=True,
        tolerance=0.00001,
        max_iters=100,
        ext_score_adjustment=0.5,
        method="cos",
        constant=0.1,
        alpha=False,
        seed=None,
    ):
        """
        Build and store the test-level summary statistics table.

        Auto-triggers fit_statistics() if not yet run. Produces a compact
        two-column table (Items, Persons) covering mean, SD, separation
        ratio, strata, and reliability for both items and persons.

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
        method : str, default 'cos'
            Priority vector extraction method for calibration.
        constant : float, default 0.1
            Additive smoothing constant for calibration.
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
                seed=seed,
            )

        items_col = [
            self.items.mean(),
            self.items.std(),
            self.isi,
            self.item_strata,
            self.item_reliability,
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
            persons_col.append(self._cronbach_alpha())
            index.append("Cronbach alpha")

        self.test_stats = pd.DataFrame(
            {"Items": items_col, "Persons": persons_col}, index=index
        )
        self.test_stats = round(self.test_stats, dp)

    def save_stats(
        self,
        filename,
        format="csv",
        dp=3,
        warm_corr=True,
        tolerance=0.00001,
        max_iters=100,
        ext_score_adjustment=0.5,
        method="cos",
        constant=0.1,
        no_of_samples=500,
        interval=None,
    ):
        """
        Export item, person, and test statistics to file.

        Auto-triggers item_stats_df(), person_stats_df(), and test_stats_df()
        if not yet run. Saves all three tables to either a single Excel
        workbook (one sheet per table) or three separate CSV files.

        Parameters
        ----------
        filename : str
            Output filename or path (without extension for CSV format;
            '.xlsx' is appended automatically if needed for Excel format).
        format : str, default 'csv'
            Output format: 'csv' saves three separate CSV files suffixed
            '_item_stats.csv', '_person_stats.csv', '_test_stats.csv'.
            'xlsx' saves all three tables to separate sheets in a single
            Excel workbook.
        dp : int, default 3
            Decimal places for rounding. Passed to the stats_df methods
            if they have not yet been run.
        warm_corr : bool, default True
            Warm bias correction. Passed to stats_df methods if needed.
        tolerance : float, default 0.00001
            Newton-Raphson convergence tolerance.
        max_iters : int, default 100
            Maximum Newton-Raphson iterations.
        ext_score_adjustment : float, default 0.5
            Extreme score adjustment for person location estimation.
        method : str, default 'cos'
            Priority vector extraction method for calibration.
        constant : float, default 0.1
            Additive smoothing constant for calibration.
        no_of_samples : int, default 500
            Bootstrap samples for SE estimation.
        interval : float or None, default None
            Confidence interval width for item SEs.

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

            if filename[-5:] != ".xlsx":
                filename += ".xlsx"

            # BUG FIX: use context manager; writer.close() is the old pattern
            # and can leave the file handle open if an exception occurs mid-write.
            with pd.ExcelWriter(filename, engine="openpyxl") as writer:
                self.item_stats.to_excel(writer, sheet_name="Item statistics")
                self.person_stats.to_excel(writer, sheet_name="Person statistics")
                self.test_stats.to_excel(writer, sheet_name="Test statistics")

        else:
            if filename[-4:] == ".csv":
                filename = filename[:-4]

            self.item_stats.to_csv(f"{filename}_item_stats.csv")
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
        method="cos",
        constant=0.1,
    ):
        """
        Export residual correlation analysis results to file.

        Auto-triggers fit_statistics() (which includes res_corr_analysis())
        if not yet run. Saves eigenvectors, eigenvalues, variance explained,
        and PCA loadings to either a single file or separate files.

        Parameters
        ----------
        filename : str
            Output filename or path (without extension for CSV; extension
            is appended automatically).
        format : str, default 'csv'
            Output format: 'csv' or 'xlsx'.
        single : bool, default True
            If True and format='csv', writes all four tables sequentially
            into a single CSV file separated by blank lines.
            If True and format='xlsx', writes all tables to one Excel
            sheet ('Item residual analysis') at successive row offsets.
            If False, writes each table to a separate file or sheet.
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
        method : str, default 'cos'
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

        if single:
            if format == "xlsx":

                if filename[-5:] != ".xlsx":
                    filename += ".xlsx"

                # BUG FIX: use context manager instead of writer.close()
                with pd.ExcelWriter(filename, engine="openpyxl") as writer:
                    row = 0
                    self.eigenvectors.round(dp).to_excel(
                        writer,
                        sheet_name="Item residual analysis",
                        startrow=row,
                        startcol=0,
                    )
                    row += self.eigenvectors.shape[0] + 2
                    self.eigenvalues.round(dp).to_excel(
                        writer,
                        sheet_name="Item residual analysis",
                        startrow=row,
                        startcol=0,
                    )
                    row += self.eigenvalues.shape[0] + 2
                    self.variance_explained.round(dp).to_excel(
                        writer,
                        sheet_name="Item residual analysis",
                        startrow=row,
                        startcol=0,
                    )
                    row += self.variance_explained.shape[0] + 2
                    self.loadings.round(dp).to_excel(
                        writer,
                        sheet_name="Item residual analysis",
                        startrow=row,
                        startcol=0,
                    )

            else:
                if filename[-4:] != ".csv":
                    filename += ".csv"

                with open(filename, "a") as f:
                    self.eigenvectors.round(dp).to_csv(f)
                    f.write("\n")
                    self.eigenvalues.round(dp).to_csv(f)
                    f.write("\n")
                    self.variance_explained.round(dp).to_csv(f)
                    f.write("\n")
                    self.loadings.round(dp).to_csv(f)

        else:
            if format == "xlsx":

                if filename[-5:] != ".xlsx":
                    filename += ".xlsx"

                # BUG FIX: use context manager instead of writer.close()
                with pd.ExcelWriter(filename, engine="openpyxl") as writer:
                    self.eigenvectors.round(dp).to_excel(
                        writer, sheet_name="Eigenvectors"
                    )
                    self.eigenvalues.round(dp).to_excel(
                        writer, sheet_name="Eigenvalues"
                    )
                    self.variance_explained.round(dp).to_excel(
                        writer, sheet_name="Variance explained"
                    )
                    self.loadings.round(dp).to_excel(
                        writer, sheet_name="Principal Component loadings"
                    )

            else:
                if filename[-4:] == ".csv":
                    filename = filename[:-4]

                self.eigenvectors.round(dp).to_csv(f"{filename}_eigenvectors.csv")
                self.eigenvalues.round(dp).to_csv(f"{filename}_eigenvalues.csv")
                self.variance_explained.round(dp).to_csv(
                    f"{filename}_variance_explained.csv"
                )
                self.loadings.round(dp).to_csv(
                    f"{filename}_principal_component_loadings.csv"
                )

    def class_intervals(self, items=None, no_of_classes=5):
        """
        Compute class interval mean person locations and mean observed scores.

        Partitions persons into quantile-based person location groups and computes
        the mean person location and mean observed score within each group. Used to
        generate observed-data overlays on TCC and ICC plots.

        Requires person_estimates() to have been run first (self.persons
        must exist).

        Parameters
        ----------
        items : list or None, default None
            Item subset to use for scoring. None uses all items. Only
            persons with complete data on the specified items are included.
        no_of_classes : int, default 5
            Number of class intervals (quantile groups) to partition
            persons into.

        Returns
        -------
        mean_person_locations : pandas.Series
            Mean person location estimate within each class interval, indexed by
            class label ('class_1', 'class_2', ...).
        obs : pandas.Series
            Mean observed total score within each class interval, indexed
            by class label.
        """

        class_groups = [f"class_{class_no + 1}" for class_no in range(no_of_classes)]

        df = self.responses.copy()

        if items is None:
            items = list(self.responses.columns)

        df = df[items].dropna(how="all")
        estimates = self.persons.loc[df.index]

        quantiles = estimates.quantile(
            [(i + 1) / no_of_classes for i in range(no_of_classes - 1)]
        )

        mask_dict = {}
        mask_dict["class_1"] = estimates < quantiles.values[0]
        mask_dict[f"class_{no_of_classes}"] = (
            estimates >= quantiles.values[no_of_classes - 2]
        )
        for class_no in range(no_of_classes - 2):
            mask_dict[f"class_{class_no + 2}"] = (
                estimates >= quantiles.values[class_no]
            ) & (estimates < quantiles.values[class_no + 1])

        mean_person_locations = {
            class_group: estimates[mask_dict[class_group]].mean()
            for class_group in class_groups
        }
        mean_person_locations = pd.Series(mean_person_locations)

        obs = {
            class_group: df[mask_dict[class_group]].mean().sum()
            for class_group in class_groups
        }

        for class_group in class_groups:
            obs[class_group] = pd.Series(obs[class_group])

        obs = pd.concat(obs, keys=obs.keys())

        return mean_person_locations, obs

    def class_intervals_cats(self, item, no_of_classes=5):
        """
        Compute class interval mean person locations and observed category proportions.

        Partitions persons into quantile-based person location groups using all items,
        then computes the proportion of 0 and 1 responses within each group
        for a specified item. Used to generate observed-data overlays on ICC
        plots for the SLM.

        Requires person_estimates() to have been run first.

        Parameters
        ----------
        item : str
            Item identifier for which to compute observed category proportions.
        no_of_classes : int, default 5
            Number of class intervals.

        Returns
        -------
        mean_person_locations : pandas.Series
            Mean person location within each class interval.
        obs_props : numpy.ndarray
            Array of shape (no_of_classes, 2) where column 0 is the
            proportion of 0 responses and column 1 is the proportion of
            1 responses in each class interval.
        """

        class_groups = [f"class_{class_no + 1}" for class_no in range(no_of_classes)]

        mean_person_locations, obs_means = self.class_intervals(
            items=[item], no_of_classes=no_of_classes
        )

        obs_props = {
            class_group: np.array(
                [1 - obs_means[class_group][0], obs_means[class_group][0]]
            )
            for class_group in class_groups
        }

        obs_props = pd.DataFrame(obs_props).to_numpy().T

        return mean_person_locations, obs_props

    """
    Plots
    """

    def plot_data(
        self,
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
        graph_title="",
        y_label="",
        plot_style="white",
        palette="dark blue",
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
        Core plotting engine for all SLM item and test characteristic curves.

        Renders one or more curves (y_data columns) against an x-axis
        (typically person location), with optional observed-data overlays, score
        lines, information lines, and CSEM lines. Called internally by
        icc(), tcc(), test_info(), and test_csem(); not normally called
        directly by users.

        Parameters
        ----------
        x_data : array-like
            X-axis values (typically a fine person location grid from -20 to 20).
        y_data : numpy.ndarray
            2-D array of shape (len(x_data), n_curves) containing the
            curve values to plot. Each column is rendered as a separate line.
        x_min : float, default -5
            Left limit of the displayed x-axis.
        x_max : float, default 5
            Right limit of the displayed x-axis.
        y_max : float, default 0
            Upper limit of the y-axis. If 0, matplotlib auto-scales.
        items : list or None, default None
            Item subset, used to look up item locations for score lines
            and threshold lines.
        obs : bool, default False
            If True, plots observed data overlays using x_obs_data and
            y_obs_data.
        x_obs_data : numpy.ndarray, default empty
            X coordinates of observed data points.
        y_obs_data : numpy.ndarray, default empty
            Y coordinates of observed data points, shape (n_points, n_curves).
        thresh_line : bool, default False
            If True, draws a vertical dashed line at the item location.
        score_lines_item : list, default [None, None]
            [item_name, list_of_proportions] for item-level score lines.
            Draws vertical and horizontal dashed lines at each proportion.
        score_lines_test : list or None, default None
            List of raw scores for test-level score lines on the TCC.
        point_info_lines_item : list, default [None, None]
            Item-level information lines (not currently used by SLM methods).
        point_info_lines_test : list or None, default None
            Test-level information lines on the test information curve.
        point_csem_lines : list or None, default None
            CSEM values at which to draw horizontal reference lines.
        score_labels : bool, default False
            If True, annotates score line intersections with numeric values.
        point_info_labels : bool, default False
            If True, annotates information line intersections.
        warm : bool, default True
            Unused directly in this method; passed for API consistency.
        cat_highlight : int or None, default None
            Category to highlight (not used in SLM; relevant in RSM/PCM).
        graph_title : str, default ''
            Plot title string.
        y_label : str, default ''
            Y-axis label string.
        plot_style : str, default 'white'
            Seaborn style: 'white' (whitegrid) or 'dark' (darkgrid).
        palette : str, default 'dark blue'
            Colour palette name. Options: 'dark blue', 'light blue',
            'dark red', 'light red', 'dark green', 'light green',
            'dark grey', 'light grey', 'dark multi', 'light multi'.
        black : bool, default False
            If True, all curves are plotted in black regardless of palette.
        figsize : tuple, default (8, 6)
            Figure size in inches (width, height).
        font : str, default 'Times New Roman'
            Font family for plot text.
        title_font_size : int, default 15
            Font size for the plot title.
        axis_font_size : int, default 12
            Font size for axis labels.
        labelsize : int, default 12
            Font size for tick labels.
        tex : bool, default True
            If True, attempts to use LaTeX rendering for text.
        plot_density : int, default 300
            Output DPI when saving to file.
        filename : str or None, default None
            If provided, saves the plot to this path. Format determined
            by file_format.
        file_format : str, default 'png'
            File format for saved plots (e.g. 'png', 'pdf', 'svg').

        Returns
        -------
        matplotlib.figure.Figure
            The rendered matplotlib Figure object.
        """

        # BUG FIX: the original tex block was a no-op —
        # str.join() returns a new string but the result was never assigned,
        # so LaTeX packages were never actually loaded. Removed entirely.
        # Font is set via fontname= on each text element below.

        if plot_style == "dark":
            sns.set_style("darkgrid")

        else:
            sns.set_style("whitegrid")

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
        }

        if palette_dict[palette][0] == "dark":
            if palette == "dark multi":
                color_map = sns.color_palette("dark", as_cmap=True)
            else:
                color_map = sns.dark_palette(
                    palette_dict[palette][1], reverse=True, as_cmap=True
                )

        if palette_dict[palette][0] == "light":
            if palette == "light multi":
                color_map = sns.color_palette("muted", as_cmap=True)
            else:
                color_map = sns.light_palette(
                    palette_dict[palette][1], reverse=True, as_cmap=True
                )

        graph, ax = plt.subplots(figsize=figsize)

        no_of_plots = y_data.shape[1]

        cNorm = colors.Normalize(vmin=0, vmax=no_of_plots + 2)

        if "multi" not in palette:
            scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=color_map)

        if black:
            for i in range(no_of_plots):
                ax.plot(x_data, y_data[:, i], "", label=i + 1, color="black")

        else:
            for i in range(no_of_plots):
                if "multi" not in palette:
                    colorVal = scalarMap.to_rgba(i)
                else:
                    colorVal = color_map[i]

                ax.plot(x_data, y_data[:, i], "", color=colorVal, label=i + 1)

        if obs:
            if y_obs_data.ndim == 1:
                colorVal = (
                    scalarMap.to_rgba(0) if "multi" not in palette else color_map[0]
                )
                ax.scatter(
                    x_obs_data, y_obs_data, color=colorVal, s=40, alpha=0.7,
                    edgecolors="k",
                )
            else:
                no_of_obs_plots = y_obs_data.shape[1]
                for j in range(no_of_obs_plots):
                    if "multi" not in palette:
                        colorVal = scalarMap.to_rgba(j)
                    else:
                        colorVal = color_map[j]
                    ax.scatter(
                        x_obs_data, y_obs_data[:, j], color=colorVal, s=40,
                        alpha=0.7, edgecolors="k",
                    )

        if items is not None:
            item_locations = self.items.loc[items]

        else:
            item_locations = self.items

        if thresh_line:
            plt.axvline(x=self.items.loc[items], color="darkred", linestyle="--")

        if score_lines_item[1] is not None:

            if all(x > 0 for x in score_lines_item[1]) & all(
                x < 1 for x in score_lines_item[1]
            ):

                estimates_set = [
                    np.log(score) - np.log(1 - score) + self.items[items]
                    for score in score_lines_item[1]
                ]

                for thresh, estimate in zip(score_lines_item[1], estimates_set):
                    plt.vlines(
                        x=estimate,
                        ymin=-100,
                        ymax=thresh,
                        color="black",
                        linestyles="dashed",
                    )
                    if score_labels:
                        plt.text(
                            estimate + (x_max - x_min) / 100,
                            y_max / 50,
                            str(round(estimate, 2)),
                        )
                    plt.hlines(
                        y=thresh,
                        xmin=-100,
                        xmax=estimate,
                        color="black",
                        linestyles="dashed",
                    )
                    if score_labels:
                        plt.text(
                            x_min + (x_max - x_min) / 100,
                            thresh + y_max / 50,
                            str(thresh),
                        )

            else:
                warnings.warn(
                    "Invalid score for score line: values must be "
                    "strictly between 0 and 1.",
                    UserWarning,
                    stacklevel=2,
                )

        if score_lines_test is not None:

            if items is None:
                no_of_items = self.no_of_items

            else:
                if isinstance(items, list):
                    no_of_items = len(items)

                else:
                    no_of_items = 1

            if all(x > 0 for x in score_lines_test) & all(
                x < no_of_items for x in score_lines_test
            ):

                if items is None:
                    estimates_set = [
                        self.score_lookup(
                            score, items=self.responses.columns, warm_corr=False
                        )
                        for score in score_lines_test
                    ]

                else:
                    estimates_set = [
                        self.score_lookup(score, items=items, warm_corr=False)
                        for score in score_lines_test
                    ]

                for thresh, estimate in zip(score_lines_test, estimates_set):
                    plt.vlines(
                        x=estimate,
                        ymin=-100,
                        ymax=thresh,
                        color="black",
                        linestyles="dashed",
                    )
                    if score_labels:
                        plt.text(
                            estimate + (x_max - x_min) / 100,
                            y_max / 50,
                            str(round(estimate, 2)),
                        )
                    plt.hlines(
                        y=thresh,
                        xmin=-100,
                        xmax=estimate,
                        color="black",
                        linestyles="dashed",
                    )
                    if score_labels:
                        plt.text(
                            x_min + (x_max - x_min) / 100,
                            thresh + y_max / 50,
                            str(thresh),
                        )

            else:
                warnings.warn(
                    "Invalid score for score line: values must be "
                    "strictly between 0 and the number of items.",
                    UserWarning,
                    stacklevel=2,
                )

        if point_info_lines_item[1] is not None:

            item = point_info_lines_item[0]

            info_set = [
                self.variance(person_location, self.items[item])
                for person_location in point_info_lines_item[1]
            ]

            for estimate, info in zip(point_info_lines_item[1], info_set):
                plt.vlines(
                    x=estimate, ymin=-100, ymax=info, color="black", linestyles="dashed"
                )
                if point_info_labels:
                    plt.text(
                        estimate + (x_max - x_min) / 100,
                        y_max / 50,
                        str(round(estimate, 2)),
                    )
                plt.hlines(
                    y=info, xmin=-100, xmax=estimate, color="black", linestyles="dashed"
                )
                if point_info_labels:
                    plt.text(
                        x_min + (x_max - x_min) / 100,
                        info + y_max / 50,
                        str(round(info, 3)),
                    )

        if point_info_lines_test is not None:

            diffs_arr = item_locations.values
            p = self.exp_score(
                np.array(point_info_lines_test)[:, None], diffs_arr[None, :]
            )
            info_set = (p * (1 - p)).sum(axis=1)

            for estimate, info in zip(point_info_lines_test, info_set):
                plt.vlines(
                    x=estimate, ymin=-100, ymax=info, color="black", linestyles="dashed"
                )
                if point_info_labels:
                    plt.text(
                        estimate + (x_max - x_min) / 100,
                        y_max / 50,
                        str(round(estimate, 2)),
                    )
                plt.hlines(
                    y=info, xmin=-100, xmax=estimate, color="black", linestyles="dashed"
                )
                if point_info_labels:
                    plt.text(
                        x_min + (x_max - x_min) / 100,
                        info + y_max / 50,
                        str(round(info, 3)),
                    )

        if point_csem_lines is not None:

            diffs_arr = item_locations.values
            p = self.exp_score(np.array(point_csem_lines)[:, None], diffs_arr[None, :])
            info_set = (p * (1 - p)).sum(axis=1)

            info_set = np.array(info_set)
            csem_set = 1 / (info_set**0.5)

            for estimate, csem in zip(point_csem_lines, csem_set):
                plt.vlines(
                    x=estimate, ymin=-100, ymax=csem, color="black", linestyles="dashed"
                )
                if score_labels:
                    plt.text(
                        estimate + (x_max - x_min) / 100,
                        y_max / 50,
                        str(round(estimate, 2)),
                    )
                plt.hlines(
                    y=csem, xmin=-100, xmax=estimate, color="black", linestyles="dashed"
                )
                if score_labels:
                    plt.text(
                        x_min + (x_max - x_min) / 100,
                        csem + y_max / 50,
                        str(round(csem, 3)),
                    )

        if cat_highlight == 0:
            plt.axvspan(-100, self.items[items], facecolor="blue", alpha=0.2)

        elif cat_highlight == 1:
            plt.axvspan(self.items[items], 100, facecolor="blue", alpha=0.2)

        if y_max <= 0:
            y_max = 1.01

        plt.xlim(x_min, x_max)
        plt.ylim(0, y_max)

        plt.xlabel(
            "Person location", fontname=font, fontsize=axis_font_size, fontweight="bold"
        )
        plt.ylabel(y_label, fontname=font, fontsize=axis_font_size, fontweight="bold")
        plt.title(
            graph_title, fontname=font, fontsize=title_font_size, fontweight="bold"
        )

        plt.grid(True)

        plt.tick_params(axis="x", labelsize=labelsize)
        plt.tick_params(axis="y", labelsize=labelsize)

        if filename is not None:
            plt.savefig(f"{filename}.{file_format}", dpi=plot_density)

        plt.close()

        return graph

    def icc(
        self,
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
        plot_style="white",
        palette="dark blue",
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
        Plot the Item Characteristic Curve (ICC) for a single item.

        Displays the modelled probability of a correct response as a function
        of person location (logistic curve). Optionally overlays observed
        class-interval proportions and reference lines.

        Parameters
        ----------
        item : str
            Item identifier to plot. Must be a column in self.responses.
        obs : bool, default False
            If True, overlays observed class-interval mean proportions
            correct as data points on the ICC curve.
        no_of_classes : int, default 5
            Number of class intervals for the observed data overlay.
        title : str or None, default None
            Plot title. If None, no title is shown.
        thresh_line : bool, default False
            If True, draws a vertical dashed line at the item location.
        score_lines : list or None, default None
            List of proportions (between 0 and 1) at which to draw
            horizontal and vertical reference lines on the ICC.
        score_labels : bool, default False
            If True, annotates score line intersections with numeric values.
        cat_highlight : int or None, default None
            Response category (0 or 1) to shade on the plot.
        xmin : float, default -5
            Left limit of the displayed person location axis.
        xmax : float, default 5
            Right limit of the displayed person location axis.
        plot_style : str, default 'white'
            Plot background style: 'white' or 'dark'.
        palette : str, default 'dark blue'
            Colour palette name (see plot_data() for options).
        black : bool, default False
            If True, renders all lines in black.
        font : str, default 'Times New Roman'
            Font family for all plot text.
        title_font_size : int, default 15
            Title font size in points.
        axis_font_size : int, default 12
            Axis label font size in points.
        labelsize : int, default 12
            Tick label font size in points.
        filename : str or None, default None
            If provided, saves the plot to this path.
        file_format : str, default 'png'
            Output file format (e.g. 'png', 'pdf', 'svg').
        dpi : int, default 300
            Output resolution in dots per inch.

        Returns
        -------
        matplotlib.figure.Figure
            The rendered ICC plot.
        """

        if obs:
            if not hasattr(self, "persons"):
                self.person_estimates(warm_corr=False)

            xobsdata, yobsdata = self.class_intervals_cats(
                item, no_of_classes=no_of_classes
            )
            yobsdata = yobsdata[:, 1]

        else:
            xobsdata = np.array(np.nan)
            yobsdata = np.array(np.nan)

        estimates = np.arange(-20, 20, 0.1)

        y = self.exp_score(estimates, self.items[item]).reshape(-1, 1)
        y = np.array(y).reshape([len(estimates), 1])

        if title is not None:
            graphtitle = title

        else:
            graphtitle = ""

        ylabel = "Expected score"

        plot = self.plot_data(
            x_data=estimates,
            y_data=y,
            x_obs_data=xobsdata,
            y_obs_data=yobsdata,
            x_min=xmin,
            x_max=xmax,
            y_max=self.max_score,
            items=item,
            y_label=ylabel,
            graph_title=graphtitle,
            obs=obs,
            thresh_line=thresh_line,
            score_lines_item=[item, score_lines],
            score_labels=score_labels,
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

        return plot

    def crcs(
        self,
        item=None,
        obs=None,
        no_of_classes=5,
        title=None,
        thresh_line=False,
        cat_highlight=None,
        xmin=-5,
        xmax=5,
        plot_style="white",
        palette="dark blue",
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
        Plot Category Response Curves (CRCs) for a single item.

        Displays the probability of each response category (0 and 1) as a
        function of person location. For the SLM these are simply P(X=0)
        and P(X=1) = 1 - P(X=0). Optionally overlays observed class-interval
        proportions.

        Parameters
        ----------
        item : str or None, default None
            Item identifier to plot. Pass None to use a mean item location of 0.
        obs : bool, list, or None, default None
            If True or 'all', overlays observed proportions for all categories.
            If a list of category indices (e.g. [0, 1]), overlays only those.
            If None or False, no overlay is shown.
        no_of_classes : int, default 5
            Number of class intervals for the observed data overlay.
        title : str or None, default None
            Plot title. If None, no title is shown.
        thresh_line : bool, default False
            If True, draws a vertical dashed line at the item location.
        cat_highlight : int or None, default None
            Category (0 or 1) to shade on the plot.
        xmin : float, default -5
            Left limit of the displayed person location axis.
        xmax : float, default 5
            Right limit of the displayed person location axis.
        plot_style : str, default 'white'
            Plot background style: 'white' or 'dark'.
        palette : str, default 'dark blue'
            Colour palette name (see plot_data() for options).
        black : bool, default False
            If True, renders all lines in black.
        font : str, default 'Times New Roman'
            Font family for all plot text.
        title_font_size : int, default 15
            Title font size in points.
        axis_font_size : int, default 12
            Axis label font size in points.
        labelsize : int, default 12
            Tick label font size in points.
        filename : str or None, default None
            If provided, saves the plot to this path.
        file_format : str, default 'png'
            Output file format.
        dpi : int, default 300
            Output resolution in dots per inch.

        Returns
        -------
        matplotlib.figure.Figure
            The rendered CRC plot.
        """

        if item == "none":
            item = None

        if obs is not None and obs is not False:
            if not hasattr(self, "persons"):
                self.person_estimates(warm_corr=False)

            xobsdata, yobsdata = self.class_intervals_cats(
                item, no_of_classes=no_of_classes
            )

            if obs != "all":
                if not all(cat in [0, 1] for cat in obs):
                    warnings.warn(
                        "Invalid 'obs' value. Valid values are None, 'all', "
                        "or a list of category indices.",
                        UserWarning,
                        stacklevel=2,
                    )
                    return

                else:
                    yobsdata = yobsdata[:, obs]

        else:
            xobsdata = np.array(np.nan)
            yobsdata = np.array(np.nan)

        estimates = np.arange(-20, 20, 0.1)

        p = self.exp_score(estimates, self.items[item] if item else 0)
        y = np.column_stack([1 - p, p])

        if title is not None:
            graphtitle = title

        else:
            graphtitle = ""

        ylabel = "Probability"

        plot = self.plot_data(
            x_data=estimates,
            y_data=y,
            x_min=xmin,
            x_max=xmax,
            y_max=1,
            x_obs_data=xobsdata,
            y_obs_data=yobsdata,
            items=item,
            graph_title=graphtitle,
            y_label=ylabel,
            obs=obs,
            thresh_line=thresh_line,
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

        return plot

    def iic(
        self,
        item,
        thresh_line=False,
        point_info_lines=None,
        point_info_labels=False,
        cat_highlight=None,
        xmin=-5,
        xmax=5,
        ymax=None,
        plot_style="white",
        palette="dark blue",
        black=False,
        title=None,
        font="Times New Roman",
        title_font_size=15,
        axis_font_size=12,
        labelsize=12,
        filename=None,
        file_format="png",
        dpi=300,
    ):
        """
        Plot the Item Information Curve (IIC) for a single item.

        Displays Fisher information P(X=1) * P(X=0) as a function of
        person location. The peak information occurs at the point where
        person location equals item location, where the value is exactly 0.25.

        Parameters
        ----------
        item : str
            Item identifier to plot.
        thresh_line : bool, default False
            If True, draws a vertical dashed line at the item location.
        point_info_lines : list or None, default None
            List of person location values at which to draw vertical and horizontal
            reference lines showing the information at those person locations.
        point_info_labels : bool, default False
            If True, annotates point information line intersections.
        cat_highlight : int or None, default None
            Category to shade (not typically used for IIC).
        xmin : float, default -5
            Left limit of the displayed person location axis.
        xmax : float, default 5
            Right limit of the displayed person location axis.
        ymax : float or None, default None
            Upper limit of the y-axis. If None, auto-scaled to 110% of peak.
        plot_style : str, default 'white'
            Plot background style: 'white' or 'dark'.
        palette : str, default 'dark blue'
            Colour palette name.
        black : bool, default False
            If True, renders the curve in black.
        title : str or None, default None
            Plot title. If None, no title is shown.
        font : str, default 'Times New Roman'
            Font family for all plot text.
        title_font_size : int, default 15
            Title font size in points.
        axis_font_size : int, default 12
            Axis label font size in points.
        labelsize : int, default 12
            Tick label font size in points.
        filename : str or None, default None
            If provided, saves the plot to this path.
        file_format : str, default 'png'
            Output file format.
        dpi : int, default 300
            Output resolution in dots per inch.

        Returns
        -------
        matplotlib.figure.Figure
            The rendered IIC plot.
        """

        estimates = np.arange(-20, 20, 0.1)

        p = self.exp_score(estimates, self.items[item])
        y = (p * (1 - p)).reshape(-1, 1)

        if ymax is None:
            ymax = float(y.max()) * 1.1

        if title is not None:
            graphtitle = title

        else:
            graphtitle = ""

        ylabel = "Fisher information"

        plot = self.plot_data(
            x_data=estimates,
            y_data=y,
            x_min=xmin,
            x_max=xmax,
            y_max=ymax,
            items=item,
            graph_title=graphtitle,
            y_label=ylabel,
            thresh_line=thresh_line,
            point_info_lines_item=[item, point_info_lines],
            point_info_labels=point_info_labels,
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

        return plot

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
        palette="dark blue",
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
        Plot the Test Characteristic Curve (TCC).

        Displays the expected total score across all (or a subset of) items
        as a function of person location. Optionally overlays observed
        class-interval mean total scores and draws reference lines at
        specified raw scores.

        Parameters
        ----------
        items : str, list, or None, default None
            Item subset to include. None or 'all' uses all items.
            Pass a single item name or list of names to restrict.
        obs : bool, default False
            If True, overlays observed class-interval mean total scores.
        no_of_classes : int, default 5
            Number of class intervals for the observed data overlay.
        title : str or None, default None
            Plot title. If None, no title is shown.
        score_lines : list or None, default None
            List of raw scores at which to draw vertical and horizontal
            reference lines on the TCC.
        score_labels : bool, default False
            If True, annotates score line intersections with numeric values.
        xmin : float, default -5
            Left limit of the displayed person location axis.
        xmax : float, default 5
            Right limit of the displayed person location axis.
        plot_style : str, default 'white'
            Plot background style: 'white' or 'dark'.
        palette : str, default 'dark blue'
            Colour palette name (see plot_data() for options).
        black : bool, default False
            If True, renders all lines in black.
        font : str, default 'Times New Roman'
            Font family for all plot text.
        title_font_size : int, default 15
            Title font size in points.
        axis_font_size : int, default 12
            Axis label font size in points.
        labelsize : int, default 12
            Tick label font size in points.
        filename : str or None, default None
            If provided, saves the plot to this path.
        file_format : str, default 'png'
            Output file format.
        dpi : int, default 300
            Output resolution in dots per inch.

        Returns
        -------
        matplotlib.figure.Figure
            The rendered TCC plot.
        """

        if isinstance(items, str):
            if items == "all":
                items = None

            elif items == "none":
                items = None

            else:
                items = [items]

        if obs:
            if not hasattr(self, "persons"):
                self.person_estimates(warm_corr=False)

            xobsdata, yobsdata = self.class_intervals(
                items=items, no_of_classes=no_of_classes
            )
            yobsdata = np.array(yobsdata).reshape(no_of_classes, 1)

        else:
            xobsdata = np.array(np.nan)
            yobsdata = np.array(np.nan)

        estimates = np.arange(-20, 20, 0.1)

        if items is None:
            diffs_arr = self.items.values
            p = 1.0 / (1.0 + np.exp(diffs_arr[None, :] - estimates[:, None]))
            y = p.sum(axis=1, keepdims=True)

        else:
            diffs_arr = self.items[items].values
            p = 1.0 / (1.0 + np.exp(diffs_arr[None, :] - estimates[:, None]))
            y = p.sum(axis=1, keepdims=True)

        y = np.array(y).reshape(len(estimates), 1)

        if items is None:
            y_max = self.no_of_items

        else:
            y_max = len(items)

        if title is not None:
            graphtitle = title

        else:
            graphtitle = ""

        ylabel = "Expected score"

        plot = self.plot_data(
            x_data=estimates,
            y_data=y,
            items=items,
            x_obs_data=xobsdata,
            y_obs_data=yobsdata,
            x_min=xmin,
            x_max=xmax,
            y_max=y_max,
            score_lines_test=score_lines,
            graph_title=graphtitle,
            y_label=ylabel,
            obs=obs,
            score_labels=score_labels,
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

        return plot

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
        palette="dark blue",
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

        Displays the sum of item Fisher information values across all (or a
        subset of) items as a function of person location. Higher information
        indicates greater measurement precision at that person location level.

        Parameters
        ----------
        items : str, list, or None, default None
            Item subset to include. None or 'all' uses all items.
        point_info_lines : list or None, default None
            List of person location values at which to draw vertical and horizontal
            reference lines showing the total information at those person locations.
        point_info_labels : bool, default False
            If True, annotates point information line intersections.
        xmin : float, default -5
            Left limit of the displayed person location axis.
        xmax : float, default 5
            Right limit of the displayed person location axis.
        ymax : float or None, default None
            Upper limit of the y-axis. If None, auto-scaled to 110% of peak.
        title : str or None, default None
            Plot title. If None, no title is shown.
        plot_style : str, default 'white'
            Plot background style: 'white' or 'dark'.
        palette : str, default 'dark blue'
            Colour palette name.
        black : bool, default False
            If True, renders all lines in black.
        font : str, default 'Times New Roman'
            Font family for all plot text.
        title_font_size : int, default 15
            Title font size in points.
        axis_font_size : int, default 12
            Axis label font size in points.
        labelsize : int, default 12
            Tick label font size in points.
        filename : str or None, default None
            If provided, saves the plot to this path.
        file_format : str, default 'png'
            Output file format.
        dpi : int, default 300
            Output resolution in dots per inch.

        Returns
        -------
        matplotlib.figure.Figure
            The rendered test information curve plot.
        """

        if isinstance(items, str):
            if items == "all":
                items = None

            elif items == "none":
                items = None

            else:
                items = [items]

        estimates = np.arange(-20, 20, 0.1)

        if items is None:
            diffs_arr = self.items.values
            p = 1.0 / (1.0 + np.exp(diffs_arr[None, :] - estimates[:, None]))
            y = (p * (1.0 - p)).sum(axis=1, keepdims=True)

        else:
            diffs_arr = self.items[items].values
            p = 1.0 / (1.0 + np.exp(diffs_arr[None, :] - estimates[:, None]))
            y = (p * (1.0 - p)).sum(axis=1, keepdims=True)

        y = np.array(y).reshape(len(estimates), 1)

        if ymax is None:
            ymax = float(y.max()) * 1.1

        if title is not None:
            graphtitle = title

        else:
            graphtitle = ""

        ylabel = "Fisher information"

        plot = self.plot_data(
            x_data=estimates,
            y_data=y,
            items=items,
            x_min=xmin,
            x_max=xmax,
            y_max=ymax,
            graph_title=graphtitle,
            point_info_lines_test=point_info_lines,
            point_info_labels=point_info_labels,
            y_label=ylabel,
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

        return plot

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
        palette="dark blue",
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

        Displays 1 / sqrt(I(theta)) — the conditional standard error of
        measurement as a function of person location — where I(theta) is the
        total test information. Lower CSEM indicates greater measurement
        precision. Optionally draws reference lines at specified person location values.

        Parameters
        ----------
        items : str, list, or None, default None
            Item subset to include. None or 'all' uses all items.
        point_csem_lines : list or None, default None
            List of person location values at which to draw vertical and horizontal
            reference lines showing the CSEM at those person locations.
        point_csem_labels : bool, default False
            If True, annotates CSEM line intersections with numeric values.
        xmin : float, default -5
            Left limit of the displayed person location axis.
        xmax : float, default 5
            Right limit of the displayed person location axis.
        ymax : float, default 5
            Upper limit of the y-axis.
        title : str or None, default None
            Plot title. If None, no title is shown.
        plot_style : str, default 'white'
            Plot background style: 'white' or 'dark'.
        palette : str, default 'dark blue'
            Colour palette name.
        black : bool, default False
            If True, renders all lines in black.
        font : str, default 'Times New Roman'
            Font family for all plot text.
        title_font_size : int, default 15
            Title font size in points.
        axis_font_size : int, default 12
            Axis label font size in points.
        labelsize : int, default 12
            Tick label font size in points.
        filename : str or None, default None
            If provided, saves the plot to this path.
        file_format : str, default 'png'
            Output file format.
        dpi : int, default 300
            Output resolution in dots per inch.

        Returns
        -------
        matplotlib.figure.Figure
            The rendered CSEM curve plot.
        """

        if isinstance(items, str):
            if items == "all":
                items = None

            elif items == "none":
                items = None

            else:
                items = [items]

        estimates = np.arange(-20, 20, 0.1)

        if items is None:
            diffs_arr = self.items.values
            p = 1.0 / (1.0 + np.exp(diffs_arr[None, :] - estimates[:, None]))
            y = (p * (1.0 - p)).sum(axis=1, keepdims=True)

        else:
            diffs_arr = self.items[items].values
            p = 1.0 / (1.0 + np.exp(diffs_arr[None, :] - estimates[:, None]))
            y = (p * (1.0 - p)).sum(axis=1, keepdims=True)

        y = 1 / (y**0.5)
        y = y.reshape(len(estimates), 1)

        if title is not None:
            graphtitle = title

        else:
            graphtitle = ""

        ylabel = "Conditional SEM"

        plot = self.plot_data(
            x_data=estimates,
            y_data=y,
            items=items,
            x_min=xmin,
            x_max=xmax,
            y_max=ymax,
            graph_title=graphtitle,
            point_csem_lines=point_csem_lines,
            score_labels=point_csem_labels,
            y_label=ylabel,
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

        return plot

    def std_residuals_plot(
        self,
        items=None,
        bin_width=0.5,
        x_min=-5,
        x_max=5,
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
        Optionally overlays a standard normal density curve for comparison.

        Requires fit_statistics() to have been run first (self.std_residual_df
        must exist).

        Parameters
        ----------
        items : str, list, or None, default None
            Item subset to include. None or 'all' uses all items.
            Pass a single item name string or list of names to restrict.
        bin_width : float, default 0.5
            Width of histogram bins in standardised residual units.
        x_min : float, default -5
            Left limit of the displayed x-axis.
        x_max : float, default 5
            Right limit of the displayed x-axis.
        normal : bool, default False
            If True, overlays a standard normal density curve on the histogram.
        title : str or None, default None
            Plot title. If None, no title is shown.
        plot_style : str, default 'white'
            Plot background style: 'white' or 'dark'.
        black : bool, default False
            If True, renders the histogram in black.
        font : str, default 'Times New Roman'
            Font family for all plot text.
        title_font_size : int, default 15
            Title font size in points.
        axis_font_size : int, default 12
            Axis label font size in points.
        labelsize : int, default 12
            Tick label font size in points.
        filename : str or None, default None
            If provided, saves the plot to this path.
        file_format : str, default 'png'
            Output file format.
        plot_density : int, default 300
            Output resolution in dots per inch.

        Returns
        -------
        matplotlib.figure.Figure
            The rendered standardised residuals histogram.
        """

        if isinstance(items, str):
            if items == "all":
                items = None

            elif items == "none":
                items = None

            else:
                items = [items]

        if items is None:
            std_residual_df = self.std_residual_df

        else:
            std_residual_df = self.std_residual_df[items]

        std_residual_list = std_residual_df.unstack().dropna()

        plot = self.std_residuals_hist(
            std_residual_list,
            bin_width=bin_width,
            x_min=x_min,
            x_max=x_max,
            normal=normal,
            title=title,
            plot_style=plot_style,
            font=font,
            title_font_size=title_font_size,
            axis_font_size=axis_font_size,
            labelsize=labelsize,
            black=black,
            filename=filename,
            file_format=file_format,
            plot_density=plot_density,
        )

        return plot
