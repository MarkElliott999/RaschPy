import warnings
import numpy as np
import pandas as pd
from collections import deque
from matplotlib import pyplot as plt
import seaborn as sns
from math import floor
from scipy.stats import norm

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

    pass


class Rasch:
    """
    Abstract base class for all RaschPy model objects.

    Provides shared infrastructure used by SLM, PCM, RSM, and MFRM:
    data connectivity validation, priority vector extraction for PAIR
    calibration, point-measure correlation computation, item and person
    rename utilities, and the standardised residuals histogram.

    Not intended to be instantiated directly — use the concrete subclasses
    SLM, PCM, RSM, or MFRM instead.
    """

    def __init__(self):
        pass

    def check_data_connectivity(self):
        """
        Validate whether the item response network is fully connected.

        Constructs an undirected adjacency graph where two items are connected
        if at least one person responded to both. Uses BFS to find connected
        components. A disconnected network means item difficulties are estimated
        independently per component and cannot be placed on a common scale.

        Called automatically by SLM.__init__() when validate=True.

        Returns
        -------
        dict
            Always contains:
            - 'connected' : bool — True if the network is fully connected.
            - 'components_count' : int — number of connected components.
            If disconnected, also contains:
            - 'isolated_items' : list — items forming singleton components.
            - 'all_sub_groups' : list of lists — all components.
        """
        if not hasattr(self, "responses") or self.responses is None:
            return {"connected": False, "reason": "No responses loaded."}

        df_array = np.array(self.responses, dtype=np.float64)
        item_names = list(self.responses.columns)
        no_of_items = len(item_names)

        if no_of_items == 0:
            return {"connected": False, "reason": "No items present in the responses."}

        # Vectorized paired comparisons matrix (ignoring NaNs)
        is_one = (df_array == 1) & (~np.isnan(df_array))
        is_zero = (df_array == 0) & (~np.isnan(df_array))
        raw_matrix = np.dot(is_one.T, is_zero).astype(np.float64)

        # Undirected adjacency matrix
        adjacency = ((raw_matrix + raw_matrix.T) > 0).astype(np.int8)

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
            return {"connected": True, "components_count": 1}
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
            }

    def priority_vector(
        self, matrix, method="cos", log_lik_tol=0.000001, pcm=False, raters=False
    ):
        """
        Extract a priority vector (item difficulty estimates) from a pairwise matrix.

        Implements the Choppin (1968) PAIR algorithm: given a matrix where
        entry (i, j) counts persons who passed item i and failed item j, extracts
        a log-scale priority vector proportional to item difficulty. Three methods
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
            'evm'      — eigenvector method via PCA. Requires scikit-learn.
        log_lik_tol : float, default 0.000001
            Convergence tolerance for the 'log-lik' method.
        pcm : bool, default False
            If True, names output using item-threshold labels for PCM calibration.
        raters : bool, default False
            If True, names output using rater labels for MFRM calibration.

        Returns
        -------
        pandas.Series
            Item difficulty (or rater severity) estimates, zero-centred logits,
            indexed by item (or rater) name. Returns None if 'evm' fails.
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
            # PCA was referenced but never imported in the original code.
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
        item scores and person ability estimates. Expected point-measure
        correlation uses modelled expected scores corrected for shrinkage.

        Parameters
        ----------
        abils : pandas.Series
            Person ability estimates indexed by person identifier.
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

        # Use .notna() for the validity mask — cleaner and avoids
        # division-by-zero artifacts from the original (x+1)/(x+1) approach.
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

    def rename_persons_all(self, new_names):
        """
        Rename all persons at once.

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
            self.responses.rename(
                index=dict(zip(self.responses.index, new_names)), inplace=True
            )

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
        Plot a histogram of standardised residuals with an optional Normal overlay.

        Shared implementation called by std_residuals_plot() in SLM, PCM, RSM,
        and MFRM. Under a well-fitting Rasch model, standardised residuals should
        approximate a standard normal distribution.

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
            If provided, saves the plot to this path.
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
