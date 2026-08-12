import warnings
import numpy as np
import pandas as pd
from collections import deque
from matplotlib import pyplot as plt
import seaborn as sns
from math import floor
from scipy.stats import norm, t as t_dist

# PCA import guarded — only needed for the 'evm' priority_vector method.
try:
    from sklearn.decomposition import PCA
except ImportError:
    PCA = None

# Apply base style once at module load rather than on every plot call.
# Individual methods can override via plt.rc_context() if needed.
plt.style.use("seaborn-v0_8-white")
sns.set_style("whitegrid")


class _SimParams:
    """
    Namespace container for generating parameters when a model is instantiated
    from a simulation object. Accessible via model.generating.

    All attributes of the sim object are copied here, so generating parameters
    (e.g. model.generating.persons, model.generating.thresholds) are preserved
    separately from the model's fitted estimates, even when person counts differ
    due to invalid/extreme response removal.
    """

    def __getattr__(self, name):
        """
        Alternate spellings for lambda_/omega (and, for bistretch,
        omega_items/omega_thresholds), matching the same aliases the
        originating sim object itself supports (e.g.
        MFRM_Sim_Centrality.__getattr__) -- 'global_' for 'lambda_',
        'stretch'/'stretch_items'/'stretch_thresholds' for
        'omega'/'omega_items'/'omega_thresholds'. setattr-based copying
        (see MFRM.__init__'s sim-aware instantiation) only transfers the
        sim's real instance attributes, not its own __getattr__-resolved
        aliases, so this namespace needs the same alias resolution
        applied again here -- generically, so it works regardless of
        which sim type populated it. Raises AttributeError (rather than
        a wrong/misleading value) if the underlying attribute was never
        copied over. Only triggers when normal attribute lookup fails,
        so it never shadows a real attribute.
        """
        aliases = {
            "global_": "lambda_",
            "stretch": "omega",
            "stretch_items": "omega_items",
            "stretch_thresholds": "omega_thresholds",
        }
        if name in aliases and aliases[name] in vars(self):
            return getattr(self, aliases[name])
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {name!r}"
        )


class _Namespace:
    """
    General-purpose namespace container for storing several named results
    side by side (e.g. model.counts.group_a, model.counts.group_b), rather
    than a single attribute that gets overwritten on every call.

    Supports both attribute access (namespace.key) for identifier-like
    string keys, and item access (namespace[key]) for any hashable key
    (e.g. an integer, or a string that isn't a valid Python identifier).
    Both forms read/write the same underlying storage.
    """

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return f"{self.__class__.__name__}({self.__dict__!r})"


class Rasch:
    """
    Abstract base class for all RaschPy model objects.

    Provides shared infrastructure used by SLM, PCM, RSM, and MFRM:
    data connectivity validation, priority vector extraction for PAIR
    calibration, point-measure correlation computation, item and person
    rename utilities, and the standardised residuals histogram.

    Not intended to be instantiated directly — use one of the concrete
    subclasses SLM, PCM, RSM, and MFRM instead.

    Concrete subclasses must implement _build_pairwise_matrix(self) ->
    (matrix, row_items): the raw (unsmoothed) directed pairwise comparison
    matrix calibrate() itself uses, and a same-length array mapping each
    matrix row/column to the item name it belongs to (one-to-one for
    SLM/RSM/MFRM; many-to-one, one row per item-category, for PCM). This
    lets check_data_connectivity() predict calibrate()'s real structural
    zeroes exactly, without duplicating each model's matrix-construction
    logic.
    """

    def __init__(self):
        pass

    @property
    def item_ids(self):
        """Alias for self.item_names."""
        return self.item_names

    @property
    def person_ids(self):
        """Alias for self.person_names."""
        return self.person_names

    def check_data_connectivity(self):
        """
        Validate whether the item response network is fully connected.

        Builds the same directed pairwise comparison matrix each concrete
        subclass's own calibrate() uses (via self._build_pairwise_matrix(),
        which every subclass must implement) and checks it for structural
        zeroes — entries that stay zero no matter how many times the matrix
        is powered, because there is no chain of comparisons linking the
        corresponding items/categories. Connectivity here specifically means
        what matters for PAIR/CPAT calibration: at least one person's
        response differentiates between two items (e.g. scored higher on
        one than the other), not merely that some person answered both — a
        person who scores identically on two items contributes nothing to
        their relative calibration.

        For models whose matrix is indexed at a finer grain than one row
        per item (PCM: one row per item-category), rows are collapsed back
        to their owning item before the undirected BFS (Breadth-First
        Search) that finds connected components — two items are adjacent if
        ANY pair of their rows/columns has a nonzero entry in either
        direction. A disconnected network means item locations are
        estimated independently per component and cannot be placed on a
        common scale.

        Also flags "directionally isolated" items: those with at least one
        row or column in the matrix that sums to zero (but not both — an
        entirely zero row AND column means no responses at all, already
        caught as an isolated singleton by the BFS instead). Such items can
        appear connected under the undirected view but will break
        calibrate()'s directed matrix, silently producing NaN/overflow.

        Called automatically by __init__() when validate=True.

        Returns
        -------
        dict
            Always contains:
            - 'connected' : bool — True if the network is fully connected.
            - 'components_count' : int — number of connected components.
            - 'directionally_isolated_items' : list — items with a
              structurally zero row or column in calibrate()'s matrix.
              These pass the undirected BFS but will break calibrate().
            If disconnected, also contains:
            - 'isolated_items' : list — items forming singleton components.
            - 'all_sub_groups' : list of lists — all components.
        """
        if not hasattr(self, "responses") or self.responses is None:
            return {
                "connected": False,
                "reason": "No responses loaded.",
                "components_count": 0,
                "isolated_items": [],
                "all_sub_groups": [],
                "directionally_isolated_items": [],
            }

        item_names = list(self.item_names)
        no_of_items = len(item_names)

        if no_of_items == 0:
            return {
                "connected": False,
                "reason": "No items present in the responses.",
                "components_count": 0,
                "isolated_items": [],
                "all_sub_groups": [],
                "directionally_isolated_items": [],
            }

        matrix, row_items = self._build_pairwise_matrix()
        row_items = np.asarray(row_items)

        # Row/column-level structural zero check, at whatever granularity
        # the model's own matrix uses (one row per item, or — for PCM —
        # one row per item-category). XOR excludes rows/columns that are
        # both zero (no responses at all), already caught as isolated
        # singletons by the BFS below.
        row_zero = matrix.sum(axis=1) == 0
        col_zero = matrix.sum(axis=0) == 0
        structurally_zero_rows = row_zero ^ col_zero
        directionally_isolated_items = sorted(
            set(row_items[structurally_zero_rows].tolist())
        )

        if directionally_isolated_items:
            warnings.warn(
                f"{len(directionally_isolated_items)} item(s) have a structurally "
                f"unresolvable zero in calibrate()'s directed pairwise matrix: "
                f"{directionally_isolated_items}. This can silently corrupt "
                f"calibration (NaN/overflow) even when the item(s) otherwise "
                f"appear connected. Consider dropping these items or gathering "
                f"more responses before calibrating.",
                UserWarning,
                stacklevel=2,
            )

        # Collapse row-level (possibly sub-item) adjacency to item-level
        # adjacency via a membership matrix, then run BFS over items.
        undirected = ((matrix + matrix.T) > 0).astype(np.float64)
        item_index = {item: idx for idx, item in enumerate(item_names)}
        membership = np.zeros((len(row_items), no_of_items), dtype=np.float64)
        for row_idx, row_item in enumerate(row_items):
            membership[row_idx, item_index[row_item]] = 1.0
        item_level_adjacency = (membership.T @ undirected @ membership) > 0
        np.fill_diagonal(item_level_adjacency, False)
        adjacency = item_level_adjacency.astype(np.int8)

        # Use deque for O(1) popleft instead of list.pop(0) which is O(n)
        visited = np.zeros(no_of_items, dtype=bool)
        components = []

        for item_idx in range(no_of_items):
            if not visited[item_idx]:
                component = []
                queue = deque([item_idx])
                visited[item_idx] = True

                while queue:
                    current = queue.popleft()
                    component.append(item_names[current])

                    neighbours = np.where(adjacency[current] & ~visited)[0]
                    for n in neighbours:
                        visited[n] = True
                        queue.append(n)

                components.append(component)

        is_connected = len(components) == 1

        if is_connected:
            return {
                "connected": True,
                "components_count": 1,
                "directionally_isolated_items": directionally_isolated_items,
            }
        else:
            isolated_items = [comp for comp in components if len(comp) == 1]
            sub_group_summary = "; ".join(
                f"Sub-group {i + 1} (size {len(comp)}): {comp[:5]}"
                for i, comp in enumerate(components)
            )
            warnings.warn(
                f"The data is split into {len(components)} disconnected sub-networks. "
                f"This will break un-smoothed calibrations. Isolated groupings: {sub_group_summary}",
                UserWarning,
                stacklevel=2,
            )
            return {
                "connected": False,
                "components_count": len(components),
                "isolated_items": isolated_items,
                "all_sub_groups": components,
                "directionally_isolated_items": directionally_isolated_items,
            }

    def priority_vector(
        self, matrix, method="cos", log_lik_tol=0.000001, pcm=False, raters=False
    ):
        """
        Extract a priority vector (item location estimates) from a pairwise matrix.

        Implements the Choppin (1968) PAIR algorithm: given a matrix where
        entry (i, j) counts persons who passed item i and failed item j, extracts
        a log-scale priority vector proportional to item location. Several methods
        are supported, all producing zero-centred logit estimates.

        Parameters
        ----------
        matrix : numpy.ndarray
            Square pairwise comparison matrix, shape (n, n).
        method : str, default 'cos'
            Priority vector extraction method:
            'cos'      — cosine (geometric mean) normalisation. Fast and robust.
            'ls'       — least squares (row mean of reciprocal matrix).
            'log-lik'  — iterative maximum likelihood (Bradley-Terry model).
            'evm'      — eigenvector method (Garner & Engelhard, 2002)via PCA.
                         Requires scikit-learn.
        log_lik_tol : float, default 0.000001
            Convergence tolerance for the 'log-lik' method.
        pcm : bool, default False
            If True, names output using item-threshold labels for PCM calibration.
        raters : bool, default False
            If True, names output using rater labels for MFRM calibration.

        Returns
        -------
        pandas.Series
            Item (or facet effect) estimates, zero-centred logits,
            indexed by item (or facet/rater) name. Returns None if 'evm' fails.
        """
        matrix_dim = matrix.shape[0]

        if pcm:
            names = []
            for i, item in enumerate(self.responses.columns):
                for j in range(self.max_score_vector.iloc[i]):
                    names.append(f"{str(item)}_{str(j + 1)}")
        else:
            names = self.facet_names if raters else list(self.responses.columns)

        with np.errstate(divide="ignore", invalid="ignore"):
            recip_matrix = np.divide(matrix.T, matrix)
            recip_matrix = np.nan_to_num(recip_matrix, nan=1.0, posinf=1.0, neginf=1.0)

        if method == "evm":
            if PCA is None:
                raise ImportError(
                    "scikit-learn is required for the 'evm' method. "
                    "Install it with: pip install scikit-learn"
                )
            pca = PCA()
            try:
                pca.fit(recip_matrix)
                eigenvectors = np.array(pca.components_)
                measures = -np.log(abs(eigenvectors[0]))
                measures -= np.mean(measures)
                measures = pd.Series(measures.real, index=names)
            except Exception:
                warnings.warn(
                    "EVM priority vector method failed. Try another method.",
                    UserWarning,
                    stacklevel=2,
                )
                return None

        elif method == "log-lik":
            wins = matrix.sum(axis=1)
            change = 1.0
            wins_sum = wins.sum()
            weights = (
                wins / wins_sum if wins_sum > 0 else np.ones(matrix_dim) / matrix_dim
            )
            matrix_sum_sym = matrix + matrix.T

            while change > log_lik_tol:
                weight_pairs = weights[:, np.newaxis] + weights[np.newaxis, :]
                with np.errstate(divide="ignore", invalid="ignore"):
                    term_matrix = np.divide(
                        matrix_sum_sym,
                        weight_pairs,
                        out=np.zeros_like(matrix_sum_sym),
                        where=weight_pairs > 0,
                    )
                adjustment = term_matrix.sum(axis=1)
                self_term = np.divide(
                    2 * np.diagonal(matrix),
                    2 * weights,
                    out=np.zeros(matrix_dim),
                    where=weights > 0,
                )
                adjustment -= self_term

                new_weights = np.divide(
                    wins, adjustment, out=np.zeros(matrix_dim), where=adjustment > 0
                )
                new_weights_sum = new_weights.sum()
                if new_weights_sum > 0:
                    new_weights /= new_weights_sum

                change = np.max(np.abs(weights - new_weights))
                weights = new_weights

            measures = -np.log(weights)
            measures -= np.mean(measures)
            measures = pd.Series(measures, index=names)

        else:
            if method == "ls":
                weights = np.mean(recip_matrix, axis=1)
            else:
                normaliser = np.linalg.norm(recip_matrix, axis=0)
                normalised_matrix = np.divide(
                    recip_matrix.T,
                    normaliser[:, np.newaxis],
                    out=np.zeros_like(recip_matrix.T),
                    where=normaliser[:, np.newaxis] > 0,
                )
                weights = normalised_matrix.sum(axis=0)

            measures = np.log(weights)
            measures -= np.mean(measures)
            measures = pd.Series(measures, index=names)

        return measures

    def pt_meas(self, abils, exp_score_df, info_df):
        """
        Compute observed and expected point-measure correlations.

        Point-measure correlation is the Pearson correlation between observed
        item scores and person location estimates. Expected point-measure
        correlation uses modelled expected scores corrected for shrinkage.

        Parameters
        ----------
        abils : pandas.Series
            Person location estimates indexed by person identifier.
        exp_score_df : pandas.DataFrame
            Expected scores for non-extreme persons, shape (persons, items).
        info_df : pandas.DataFrame
            Fisher information values, shape (persons, items).

        Returns
        -------
        pt_measure : pandas.Series
            Observed point-measure correlations per item.
        exp_pt_measure : pandas.Series
            Expected point-measure correlations per item.
        """
        abil_dev_df = pd.DataFrame(
            np.tile(
                abils.values[:, np.newaxis] - np.mean(abils),
                (1, len(self.responses.columns)),
            ),
            index=self.responses.index,
            columns=self.responses.columns,
        )

        mask = self.responses.notna().astype(float).replace(0, np.nan)
        abil_dev_df = (abil_dev_df * mask).loc[exp_score_df.index]

        score_dev_df = self.responses.loc[exp_score_df.index] - self.responses.mean(
            axis=0
        )
        exp_score_dev_df = exp_score_df - self.responses.loc[exp_score_df.index].mean(
            axis=0
        )

        pt_measure_num = (score_dev_df * abil_dev_df).sum(axis=0)
        pt_measure_den = (
            (score_dev_df**2).sum(axis=0) * (abil_dev_df**2).sum(axis=0)
        ) ** 0.5
        pt_measure = pt_measure_num / pt_measure_den

        resp_mask = mask.loc[exp_score_df.index]
        exp_score_dev_masked = exp_score_dev_df.where(resp_mask.notna())
        info_masked = info_df.where(resp_mask.notna())

        exp_pt_measure_num = (exp_score_dev_masked * abil_dev_df).sum(axis=0)
        exp_pt_measure_den = (
            ((exp_score_dev_masked**2) + info_masked).sum(axis=0)
            * (abil_dev_df**2).sum(axis=0)
        ) ** 0.5
        exp_pt_measure = exp_pt_measure_num / exp_pt_measure_den

        return pt_measure, exp_pt_measure

    def _cronbach_alpha(self):
        """Cronbach's alpha on complete cases; warns if missing data present."""
        data = self.responses
        if data.isnull().any().any():
            warnings.warn(
                "Missing data detected: Cronbach's alpha computed on complete "
                "cases only and may be an underestimate.",
                UserWarning,
            )
            data = data.dropna()
        k = data.shape[1]
        if k < 2 or data.shape[0] < 2:
            return np.nan
        total_var = data.sum(axis=1).var(ddof=1)
        if total_var == 0:
            return np.nan
        alpha = (k / (k - 1)) * (1 - data.var(ddof=1).sum() / total_var)
        return float(alpha)

    def rename_item(self, old, new):
        """
        Rename a single item in self.responses.

        Parameters
        ----------
        old : str
            Current item name (must be a column in self.responses).
        new : str
            Desired new item name. Must be a string and not already in use.

        Returns
        -------
        None
        """
        if old not in self.responses.columns:
            warnings.warn(
                f"Old item name {old!r} not found in data.", UserWarning, stacklevel=2
            )
            return
        if not isinstance(new, str):
            warnings.warn("Item names must be strings.", UserWarning, stacklevel=2)
            return
        if old == new:
            warnings.warn(
                "New item name is the same as the old item name.",
                UserWarning,
                stacklevel=2,
            )
            return
        if new in self.responses.columns:
            warnings.warn(
                "New item name is a duplicate of an existing item name.",
                UserWarning,
                stacklevel=2,
            )
            return
        self.responses.rename(columns={old: new}, inplace=True)

    def rename_items_all(self, new_names):
        """
        Rename all items at once.

        Parameters
        ----------
        new_names : list of str
            New item names in column order. Must match item count with no duplicates.

        Returns
        -------
        None
        """
        list_length = len(new_names)
        if len(new_names) != len(set(new_names)):
            warnings.warn(
                "List of new item names contains duplicates.", UserWarning, stacklevel=2
            )
        elif list_length != self.no_of_items:
            warnings.warn(
                f"Incorrect number of item names: {list_length} provided, "
                f"{self.no_of_items} items in data.",
                UserWarning,
                stacklevel=2,
            )
        else:
            self.responses.rename(
                columns=dict(zip(self.responses.columns, new_names)), inplace=True
            )

    def rename_person(self, old, new):
        """
        Rename a single person in self.responses.

        Also updates self.person_names, and (where present, e.g. for SLM,
        PCM and RSM) re-indexes self.exogenous and self.no_exogenous_persons
        to keep them aligned. self.exogenous_only_persons is left untouched,
        as those persons do not appear in self.responses.

        Parameters
        ----------
        old : str
            Current person name (must be in self.responses.index).
        new : str
            Desired new person name. Must be a string and not already in use.

        Returns
        -------
        None
        """
        if old not in self.responses.index:
            warnings.warn(
                f"Old person name {old!r} not found in data.", UserWarning, stacklevel=2
            )
            return
        if not isinstance(new, str):
            warnings.warn("Person names must be strings.", UserWarning, stacklevel=2)
            return
        if old == new:
            warnings.warn(
                "New person name is the same as the old person name.",
                UserWarning,
                stacklevel=2,
            )
            return
        if new in self.responses.index:
            warnings.warn(
                "New person name is a duplicate of an existing person name.",
                UserWarning,
                stacklevel=2,
            )
            return
        self.responses.rename(index={old: new}, inplace=True)
        self.person_names = self.responses.index

        if getattr(self, "exogenous", None) is not None:
            self.exogenous.rename(index={old: new}, inplace=True)
        if getattr(self, "no_exogenous_persons", None) is not None:
            self.no_exogenous_persons = self.no_exogenous_persons.map(
                lambda name: new if name == old else name
            )

    def rename_persons_all(self, new_names):
        """
        Rename all persons at once.

        Also updates self.person_names, and (where present, e.g. for SLM,
        PCM and RSM) re-indexes self.exogenous and self.no_exogenous_persons
        to keep them aligned with the renamed persons. self.exogenous_only_persons
        is left untouched, as those persons do not appear in self.responses.

        Parameters
        ----------
        new_names : list of str
            New person names in index order. Must match person count with no duplicates.

        Returns
        -------
        None
        """
        list_length = len(new_names)
        if len(new_names) != len(set(new_names)):
            warnings.warn(
                "List of new person names contains duplicates.",
                UserWarning,
                stacklevel=2,
            )
        elif list_length != self.no_of_persons:
            warnings.warn(
                f"Incorrect number of person names: {list_length} provided, "
                f"{self.no_of_persons} persons in data.",
                UserWarning,
                stacklevel=2,
            )
        elif not all(isinstance(name, str) for name in new_names):
            warnings.warn("Person names must be strings.", UserWarning, stacklevel=2)
        else:
            rename_map = dict(zip(self.responses.index, new_names))
            self.responses.rename(index=rename_map, inplace=True)
            self.person_names = self.responses.index

            if getattr(self, "exogenous", None) is not None:
                self.exogenous.rename(index=rename_map, inplace=True)
            if getattr(self, "no_exogenous_persons", None) is not None:
                self.no_exogenous_persons = self.no_exogenous_persons.map(rename_map)

    def std_residuals_hist(
        self,
        std_residual_list,
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
        Plot a histogram of standardised residuals with an optional standardised
        normal distribution overlay.

        Shared implementation called by std_residuals_plot() in SLM, PCM, RSM,
        and MFRM. Under a well-fitting Rasch model, standardised residuals should
        be broadly close to a standard normal distribution (though note that for
        SLM, the density at zero must always be precisely zero, meaning that there
        will always be a bimodal distribution, approximately symmetrical around
        the y-axis).

        Parameters
        ----------
        std_residual_list : pandas.Series
            Flat Series of standardised residuals (unstacked, NaNs dropped).
        bin_width : float, default 0.5
            Width of histogram bins.
        x_min : float, default -6
            Left x-axis limit.
        x_max : float, default 6
            Right x-axis limit.
        normal : bool, default False
            If True, overlays a standard normal density curve.
        title : str or None, default None
            Plot title. If None, no title is shown.
        plot_style : str, default 'white'
            Background style: 'white' (whitegrid) or 'dark' (darkgrid).
        black : bool, default False
            If True, renders in grey with a black normal curve.
        font : str, default 'Times New Roman'
            Font family. Set via rc_context to avoid repeated findfont() calls.
        title_font_size : int, default 15
            Title font size in points.
        axis_font_size : int, default 12
            Axis label font size in points.
        labelsize : int, default 12
            Tick label font size in points.
        filename : str or None, default None
            If provided, saves the plot to this path. No file extension needed.
        file_format : str, default 'png'
            Output file format.
        plot_density : int, default 300
            Output resolution in dots per inch.

        Returns
        -------
        None
            Displays and closes the figure. Use filename to save.
        """
        color = "gray" if black else "steelblue"
        line_color = "black" if black else "maroon"
        n_bins = floor((std_residual_list.max() - std_residual_list.min()) / bin_width)

        # Apply font and any non-default style within a context so global
        # state is not permanently mutated by a single plot call.
        style_overrides = {"font.family": font, "font.size": axis_font_size}

        with plt.rc_context(style_overrides):

            # Only re-apply style sheet when caller explicitly requests a
            # non-default style, avoiding the per-call stylesheet parse cost.
            if plot_style != "white":
                plt.style.use("seaborn-v0_8-" + plot_style)
                if plot_style == "dark":
                    sns.set_style("darkgrid")
                else:
                    sns.set_style("whitegrid")

            fig, ax = plt.subplots()

            ax.hist(
                std_residual_list,
                bins=n_bins,
                range=(x_min, x_max),
                density=True,
                facecolor=color,
                alpha=0.5,
                edgecolor="black",
                linewidth=1,
            )

            if normal:
                x_norm = np.linspace(x_min, x_max, 200)
                y_norm = norm.pdf(x_norm, 0, 1)
                ax.plot(x_norm, y_norm, color=line_color)

            ax.set_xlabel(
                "Standardised residual", fontsize=axis_font_size, fontweight="bold"
            )
            ax.set_ylabel("Density", fontsize=axis_font_size, fontweight="bold")

            if title is not None:
                ax.set_title(title, fontsize=title_font_size, fontweight="bold")

            ax.tick_params(axis="x", labelsize=labelsize)
            ax.tick_params(axis="y", labelsize=labelsize)

            if filename is not None:
                fig.savefig(filename + f".{file_format}", dpi=plot_density)

            # block=False returns immediately on all interactive backends.
            # pause(0.001) gives the GUI event loop a tick to render the
            # window, matching the behaviour of other RaschPy plot methods.
            plt.show(block=False)
            plt.pause(0.001)
            plt.close(fig)

    def _robust_anchor_selection(
        self, anchor, observed, corr_tol=0.95, sd_ratio_tol=1.1, min_anchors=6
    ):
        """
        Robust, iterative anchor-item selection (purification) via the
        modified (robust) z-score (Iglewicz & Hoaglin, 1993).

        Given externally-supplied anchor values and freshly-estimated
        observed values for the same items, iteratively drops the item
        with the largest absolute modified z-score (computed once from
        the full anchor set, not recomputed as items are dropped) until
        both the Pearson correlation and the observed/anchor standard
        deviation ratio clear the supplied tolerances, or until only
        min_anchors items remain. The resulting translation constant is
        computed only from the surviving (non-outlier) items, so a
        handful of displaced anchor items cannot contaminate the shift
        applied to the rest of the scale.

        Used identically for item-bank anchoring (anchor = external bank
        locations) and for DIF purification (anchor = a reference
        group's own item estimates, observed = the focal group's).

        Parameters
        ----------
        anchor : pandas.Series
            Reference values, indexed by item name.
        observed : pandas.Series
            Freshly-calibrated values for the same items, indexed by item
            name. Only items present in both anchor and observed are used.
        corr_tol : float, default 0.95
            Minimum acceptable Pearson correlation between anchor and
            observed values on the surviving item set.
        sd_ratio_tol : float, default 1.1
            observed.std() / anchor.std() on the surviving item set must
            fall within [1 / sd_ratio_tol, sd_ratio_tol].
        min_anchors : int, default 6
            Floor on the number of surviving items. If the initial common
            item set is at or below this floor, trimming is skipped
            entirely (a UserWarning is issued) and all supplied items are
            used. If trimming starts but reaches this floor before the
            tolerance criteria are satisfied, dropping stops there (also
            with a UserWarning) — residual outliers may remain.

        Returns
        -------
        dict with keys:
            table : pandas.DataFrame
                Per-item diagnostics (Anchor, Observed, Deviation,
                Robust z, Selected), indexed by item name.
            selected_anchors, dropped_anchors : pandas.Index
                Items retained / dropped as outliers (drop order preserved).
            original_anchor_corr, original_anchor_sd_ratio : float
                Correlation / SD ratio before any trimming.
            anchor_corr, anchor_sd_ratio : float
                Correlation / SD ratio after trimming.
            tc : float
                Translation constant, mean(Anchor) - mean(Observed) over
                the selected items only. Add this to `observed` values to
                shift them onto the anchor scale.
            converged : bool
                True if the tolerances were satisfied before min_anchors
                was reached.
        """
        common = anchor.index.intersection(observed.index)
        if len(common) == 0:
            raise ValueError(
                "No items are common to both anchor and observed — cannot "
                "compute a translation constant. Check that the anchor "
                "index matches the item names in the current calibration."
            )

        table = pd.DataFrame(
            {"Anchor": anchor.loc[common], "Observed": observed.loc[common]}
        )
        table["Deviation"] = table["Observed"] - table["Anchor"]

        original_anchor_corr = table["Anchor"].corr(table["Observed"])
        original_anchor_sd_ratio = table["Observed"].std() / table["Anchor"].std()

        if len(table) <= min_anchors:
            warnings.warn(
                f"Only {len(table)} items have both an anchor and an "
                f"observed value, at or below min_anchors={min_anchors}. "
                f"Skipping robust outlier trimming; all supplied anchor "
                f"items will be used.",
                UserWarning,
                stacklevel=2,
            )
            table["Robust z"] = np.nan
            table["Selected"] = True
            tc = table["Anchor"].mean() - table["Observed"].mean()
            return {
                "table": table,
                "selected_anchors": table.index,
                "dropped_anchors": pd.Index([]),
                "original_anchor_corr": original_anchor_corr,
                "original_anchor_sd_ratio": original_anchor_sd_ratio,
                "anchor_corr": original_anchor_corr,
                "anchor_sd_ratio": original_anchor_sd_ratio,
                "tc": tc,
                "converged": False,
            }

        median_dev = table["Deviation"].median()

        if (table["Deviation"] == median_dev).all():
            raise ValueError(
                "All anchor item deviations are identical; robust "
                "z-scores are undefined (0/0) for every item. Check the "
                "anchor/observed values."
            )

        mad = (table["Deviation"] - median_dev).abs().median()
        with np.errstate(divide="ignore", invalid="ignore"):
            table["Robust z"] = 0.6745 * (table["Deviation"] - median_dev) / mad

        remaining = table.copy()
        dropped_anchors = []
        converged = True
        anchor_corr = original_anchor_corr
        anchor_sd_ratio = original_anchor_sd_ratio

        while (
            (anchor_corr < corr_tol)
            or (anchor_sd_ratio < 1 / sd_ratio_tol)
            or (anchor_sd_ratio > sd_ratio_tol)
        ):
            if len(remaining) <= min_anchors:
                converged = False
                warnings.warn(
                    f"Robust anchor selection reached min_anchors="
                    f"{min_anchors} before the correlation/SD-ratio "
                    f"tolerances were satisfied (corr={anchor_corr:.3f}, "
                    f"sd_ratio={anchor_sd_ratio:.3f}). Residual outliers "
                    f"may remain in the selected anchor set.",
                    UserWarning,
                    stacklevel=2,
                )
                break

            drop = remaining["Robust z"].abs().idxmax()
            dropped_anchors.append(drop)
            remaining = remaining.drop(drop, axis=0)

            anchor_corr = remaining["Anchor"].corr(remaining["Observed"])
            anchor_sd_ratio = remaining["Observed"].std() / remaining["Anchor"].std()

        selected_anchors = remaining.index
        table["Selected"] = table.index.isin(selected_anchors)

        tc = (
            table.loc[selected_anchors, "Anchor"].mean()
            - table.loc[selected_anchors, "Observed"].mean()
        )

        return {
            "table": table,
            "selected_anchors": selected_anchors,
            "dropped_anchors": pd.Index(dropped_anchors),
            "original_anchor_corr": original_anchor_corr,
            "original_anchor_sd_ratio": original_anchor_sd_ratio,
            "anchor_corr": anchor_corr,
            "anchor_sd_ratio": anchor_sd_ratio,
            "tc": tc,
            "converged": converged,
        }

    def _wald_anchor_selection(self, anchor, observed, se, alpha=0.05, min_anchors=6):
        """
        Wald-test-based anchor-item selection: an alternative to
        _robust_anchor_selection() for deciding which supplied anchor
        items are safe to use when computing the translation constant.

        For each candidate anchor item, tests H0: this item's deviation
        from the current (precision-weighted) translation line is zero,
        i.e. z_i = (anchor_i - (observed_i + tc)) / SE(observed_i), a
        standard Wald test (anchor_i is treated as a fixed known
        constant — no sampling error of its own — matching how anchor
        values are used everywhere else in this package).

        Uses sequential exclusion (recompute tc and re-test on the
        current remaining set at each step), not a single pass against a
        translation constant fixed from the full candidate set. This
        matters for the same reason it does in MFRM's
        check_anchor_homogeneity: tc is a precision-weighted MEAN, and is
        therefore sensitive to outliers :a single severely off anchor
        item can drag that mean far enough that other, genuinely fine
        anchor items also fail the test — collateral flagging, not real
        misfit ("artificial DIF" in Andrich's sense, here applied to
        anchor selection rather than group DIF). _robust_anchor_selection
        does not have this problem because it centres on the MEDIAN,
        which one outlier barely moves; a mean-based translation constant
        has no such protection, so this method re-centres on each
        remaining subset instead.

        Parameters
        ----------
        anchor : pandas.Series
            Reference values, indexed by item name.
        observed : pandas.Series
            Freshly-calibrated values for the same items, indexed by item
            name. Only items present in both anchor and observed are used.
        se : pandas.Series
            Bootstrap SE of `observed`, indexed by item name (same index
            as observed). This is the one thing _robust_anchor_selection
            does not need but this method does — it's a genuine
            significance test, not a descriptive-statistics trim.
        alpha : float, default 0.05
            Significance level for each item's Wald test.
        min_anchors : int, default 6
            Floor on the number of surviving items — same semantics as
            _robust_anchor_selection: skip trimming entirely (with a
            warning) if the initial common set is at or below this floor;
            stop early (also with a warning) if trimming reaches this
            floor before every remaining item clears alpha.

        Returns
        -------
        dict — same shape as _robust_anchor_selection(), so callers can
        use either interchangeably:
            table : pandas.DataFrame
                Anchor, Observed, SE, Deviation, z, p, Selected — indexed
                by item name.
            selected_anchors, dropped_anchors : pandas.Index
            original_anchor_corr, original_anchor_sd_ratio : float
                Descriptive only here (Pearson corr / SD ratio before
                trimming) — not part of the stopping rule, unlike
                _robust_anchor_selection, but kept for interface
                consistency and comparability between the two methods.
            anchor_corr, anchor_sd_ratio : float
                Same, after trimming.
            tc : float
                Precision-weighted translation constant,
                sum(w_i*(anchor_i - observed_i)) / sum(w_i) over the
                finally-selected items, w_i = 1/SE_i^2.
            converged : bool
                True if every remaining item cleared alpha before
                min_anchors was reached.
        """
        common = anchor.index.intersection(observed.index).intersection(se.index)
        if len(common) == 0:
            raise ValueError(
                "No items are common to anchor, observed, and se — cannot "
                "compute a translation constant."
            )

        table = pd.DataFrame(
            {
                "Anchor": anchor.loc[common],
                "Observed": observed.loc[common],
                "SE": se.loc[common],
            }
        )

        original_anchor_corr = table["Anchor"].corr(table["Observed"])
        original_anchor_sd_ratio = table["Observed"].std() / table["Anchor"].std()

        def _wald(current_index):
            sub = table.loc[current_index]
            w = 1.0 / sub["SE"] ** 2
            tc = float((w * (sub["Anchor"] - sub["Observed"])).sum() / w.sum())
            deviation = sub["Anchor"] - (sub["Observed"] + tc)
            z = deviation / sub["SE"]
            p = pd.Series(2 * (1 - norm.cdf(z.abs())), index=current_index)
            return tc, deviation, z, p

        if len(table) <= min_anchors:
            warnings.warn(
                f"Only {len(table)} items have both an anchor and an "
                f"observed value, at or below min_anchors={min_anchors}. "
                f"Skipping Wald-based outlier trimming; all supplied anchor "
                f"items will be used.",
                UserWarning,
                stacklevel=2,
            )
            tc, deviation, z, p = _wald(table.index)
            table["Deviation"] = deviation
            table["z"] = z
            table["p"] = p
            table["Selected"] = True
            return {
                "table": table,
                "selected_anchors": table.index,
                "dropped_anchors": pd.Index([]),
                "original_anchor_corr": original_anchor_corr,
                "original_anchor_sd_ratio": original_anchor_sd_ratio,
                "anchor_corr": original_anchor_corr,
                "anchor_sd_ratio": original_anchor_sd_ratio,
                "tc": tc,
                "converged": False,
            }

        current = list(table.index)
        dropped_anchors = []
        dropped_stats = {}  # item -> (deviation, z, p) at its own removal step
        converged = True

        while True:
            tc, deviation, z, p = _wald(current)
            if (p >= alpha).all():
                break
            if len(current) <= min_anchors:
                converged = False
                warnings.warn(
                    f"Wald-based anchor selection reached min_anchors="
                    f"{min_anchors} before every remaining item cleared "
                    f"alpha={alpha}. Residual outliers may remain in the "
                    f"selected anchor set.",
                    UserWarning,
                    stacklevel=2,
                )
                break
            worst = z.abs().idxmax()
            dropped_anchors.append(worst)
            dropped_stats[worst] = (deviation.loc[worst], z.loc[worst], p.loc[worst])
            current.remove(worst)

        selected_anchors = pd.Index(current)
        table["Deviation"] = np.nan
        table["z"] = np.nan
        table["p"] = np.nan
        table.loc[current, "Deviation"] = deviation
        table.loc[current, "z"] = z
        table.loc[current, "p"] = p
        for item, (dev_i, z_i, p_i) in dropped_stats.items():
            table.loc[item, ["Deviation", "z", "p"]] = [dev_i, z_i, p_i]
        table["Selected"] = table.index.isin(selected_anchors)

        anchor_corr = table.loc[selected_anchors, "Anchor"].corr(
            table.loc[selected_anchors, "Observed"]
        )
        anchor_sd_ratio = (
            table.loc[selected_anchors, "Observed"].std()
            / table.loc[selected_anchors, "Anchor"].std()
        )

        return {
            "table": table,
            "selected_anchors": selected_anchors,
            "dropped_anchors": pd.Index(dropped_anchors),
            "original_anchor_corr": original_anchor_corr,
            "original_anchor_sd_ratio": original_anchor_sd_ratio,
            "anchor_corr": anchor_corr,
            "anchor_sd_ratio": anchor_sd_ratio,
            "tc": tc,
            "converged": converged,
        }

    def _welch_satterthwaite(self, diff, se1, n1, se2, n2):
        """
        Welch's t-test statistic, Satterthwaite degrees of freedom, and
        two-sided p-value for a difference between two independent
        estimates, each with its own already-computed standard error
        (e.g. an item location calibrated independently in a reference
        and a focal group in dif_test) — an alternative to a plain z-test
        when group sizes are small, unequal, or the two SEs differ
        substantially, since treating the combined SE as exactly known
        (rather than itself estimated, with more uncertainty at low n) is
        a shakier assumption in exactly those cases.

        Parameters
        ----------
        diff : array-like
            Difference between the two estimates (however signed) for
            each row (item, threshold, etc.).
        se1, se2 : array-like
            Each estimate's own standard error, same shape as diff.
        n1, n2 : int
            Number of persons (or other independent sampling units) each
            SE was estimated from. Used only in the Satterthwaite
            denominator — se1/se2 are already SEs of the mean, not raw
            per-observation SDs, so they are not rescaled by n1/n2 here.

        Returns
        -------
        t_stat : ndarray
            Test statistic — numerically identical to a z-test's
            (diff / sqrt(se1**2 + se2**2)); only the reference
            distribution (and hence df and p) differs from a z-test.
        df : ndarray
            Welch-Satterthwaite degrees of freedom, same shape as diff.
        p : ndarray
            Two-sided p-value from the t-distribution at each row's own df.
        """
        se1 = np.asarray(se1, dtype=float)
        se2 = np.asarray(se2, dtype=float)
        diff = np.asarray(diff, dtype=float)
        combined_var = se1**2 + se2**2
        t_stat = diff / np.sqrt(combined_var)
        df = combined_var**2 / (se1**4 / (n1 - 1) + se2**4 / (n2 - 1))
        p = 2 * t_dist.sf(np.abs(t_stat), df)
        return t_stat, df, p

    def _resolve_andersen_groups(self, split_by, covariate, non_extreme, person_location_values, score_values):
        """
        Shared group-splitting logic for andersen_lr_test (SLM/PCM/RSM).

        Validates split_by/covariate and builds the {group_name: index}
        mapping used to split non-extreme persons into two groups, either
        by a median split on person location/score or by an exogenous
        person covariate. Identical across SLM/PCM/RSM's andersen_lr_test,
        so factored out here rather than duplicated per model.

        Parameters
        ----------
        split_by : str
            'person_location', 'score', or 'exogenous'.
        covariate : str or None
            Column name in self.exogenous. Required (and only used) when
            split_by='exogenous'.
        non_extreme : pandas.Index
            Non-extreme person index (already restricted by the caller).
        person_location_values : pandas.Series
            Person location estimates, full index — sliced to
            non_extreme internally. Only used when split_by='person_location'.
        score_values : pandas.Series
            Raw total scores, full index — sliced to non_extreme
            internally. Only used when split_by='score'.

        Returns
        -------
        dict {group_name: pandas.Index}
            Two entries: 'low'/'high' for split_by='person_location'/'score',
            or the two observed covariate values (as strings) for
            split_by='exogenous'.
        """
        if split_by not in ("person_location", "score", "exogenous"):
            raise ValueError("split_by must be 'person_location', 'score', or 'exogenous'")

        if split_by == "exogenous":
            if covariate is None:
                raise ValueError("covariate must be specified when split_by='exogenous'")
            if getattr(self, "exogenous", None) is None:
                raise ValueError(
                    "No exogenous data available. Pass exogenous= to the "
                    "constructor before calling andersen_lr_test(split_by='exogenous')."
                )
            if covariate not in self.exogenous.columns:
                raise ValueError(f"'{covariate}' is not a column in self.exogenous.")

            cov_values = self.exogenous.loc[non_extreme, covariate].dropna()
            levels = sorted(cov_values.unique(), key=str)
            if len(levels) != 2:
                raise ValueError(
                    f"andersen_lr_test with split_by='exogenous' requires exactly 2 "
                    f"distinct non-null values for '{covariate}' among non-extreme "
                    f"persons, found {len(levels)}. Use dif_test() for covariates "
                    f"with more than two levels."
                )
            return {
                str(level): cov_values.index[cov_values == level] for level in levels
            }

        split_var = (person_location_values if split_by == "person_location" else score_values).loc[non_extreme]
        median_val = split_var.median()
        return {
            "low": split_var.index[split_var < median_val],
            "high": split_var.index[split_var > median_val],
        }

    def _bh_correction(self, pvalues):
        """
        Benjamini-Hochberg adjusted p-values (step-up FDR procedure).

        Parameters
        ----------
        pvalues : pandas.Series
            Raw p-values, indexed by item (or other comparison unit) name.

        Returns
        -------
        pandas.Series
            BH-adjusted p-values, same index as pvalues, capped at 1 and
            enforced monotonic (an adjusted p-value can never be smaller
            than that of a comparison with a smaller raw p-value).
        """
        n = len(pvalues)
        order = np.argsort(pvalues.values)
        ranked = pvalues.values[order]
        adjusted = ranked * n / (np.arange(n) + 1)
        # Step-up monotonicity: running minimum from the largest rank down
        adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
        adjusted = np.clip(adjusted, 0, 1)

        result = np.empty(n)
        result[order] = adjusted
        return pd.Series(result, index=pvalues.index)

    def plot_anchor_selection(
        self,
        result,
        xmin=-4,
        xmax=4,
        ymin=-3,
        ymax=3,
        figsize=(8, 6),
        font="Times New Roman",
        title_font_size=15,
        axis_font_size=12,
        labelsize=12,
        title="",
        filename=None,
        file_format="png",
        dpi=300,
    ):
        """
        Plot anchor vs. observed values from _robust_anchor_selection().

        Scatters selected (retained) and dropped (outlier) anchor items in
        different colours, with a solid y=x identity line and a dotted
        OLS best-fit line through the selected points only, so the two
        references are visually distinct.

        Parameters
        ----------
        result : dict or pandas.DataFrame
            Either the full return value of _robust_anchor_selection(), or
            just its 'table' entry directly (e.g. self.anchor_selection or
            an entry of self.dif_anchor_selection) — a DataFrame with
            'Anchor', 'Observed', and boolean 'Selected' columns. Selected/
            dropped item sets are derived from the 'Selected' column in
            the latter case.
        xmin, xmax, ymin, ymax : float
            Axis limits.
        figsize : tuple, default (8, 6)
            Figure size in inches (width, height).
        font : str, default 'Times New Roman'
            Font family for plot text.
        title_font_size, axis_font_size, labelsize : int
            Font sizes for title, axis labels, and tick labels.
        title : str, default ''
            Plot title string.
        filename : str or None, default None
            If provided, saves the plot to this path.
        file_format : str, default 'png'
            File format for saved plots (e.g. 'png', 'pdf', 'svg').
        dpi : int, default 300
            Output DPI when saving to file.

        Returns
        -------
        matplotlib.figure.Figure
            The rendered figure, for later inspection or saving (e.g.
            fig.savefig(...)) without needing to redraw the plot.
        """
        if isinstance(result, pd.DataFrame):
            table = result
            selected = table.index[table["Selected"]]
            dropped = table.index[~table["Selected"]]
        else:
            table = result["table"]
            selected = result["selected_anchors"]
            dropped = result["dropped_anchors"]

        fig, ax = plt.subplots(figsize=figsize)
        ax.set_aspect("equal")

        ax.scatter(
            table.loc[selected, "Anchor"],
            table.loc[selected, "Observed"],
            s=40,
            alpha=0.7,
            edgecolors="k",
            label="Selected",
        )
        if len(dropped) > 0:
            ax.scatter(
                table.loc[dropped, "Anchor"],
                table.loc[dropped, "Observed"],
                s=40,
                alpha=0.7,
                edgecolors="k",
                label="Dropped",
            )

        xseq = np.linspace(xmin, xmax, 100)
        ax.plot(xseq, xseq, color="black", lw=1, label="Identity")

        if len(selected) >= 2:
            b, a = np.polyfit(
                table.loc[selected, "Anchor"], table.loc[selected, "Observed"], deg=1
            )
            ax.plot(xseq, a + b * xseq, color="darkred", lw=1.2, linestyle=":")

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel(
            "Anchor", fontsize=axis_font_size, fontweight="bold", fontname=font
        )
        ax.set_ylabel(
            "Observed",
            fontsize=axis_font_size,
            fontweight="bold",
            fontname=font,
        )
        if title:
            ax.set_title(title, fontsize=title_font_size, fontweight="bold")
        ax.tick_params(axis="x", labelsize=labelsize)
        ax.tick_params(axis="y", labelsize=labelsize)
        ax.legend()

        if filename is not None:
            fig.savefig(f"{filename}.{file_format}", dpi=dpi)

        plt.show(block=False)
        plt.pause(0.001)

        return fig
