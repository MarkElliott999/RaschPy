import colorsys
import re
import warnings
from contextlib import contextmanager
import numpy as np
import pandas as pd
from collections import deque
from matplotlib import pyplot as plt
from matplotlib import colors as colors
from matplotlib import cm as cmx
from matplotlib.patches import Rectangle
from matplotlib.textpath import TextPath
import seaborn as sns
from math import floor
from scipy.stats import norm, t as t_dist, gaussian_kde, chi2 as chi2_dist, f as f_dist
from scipy.optimize import brentq
from scipy.sparse.csgraph import connected_components

# PCA import guarded — only needed for the 'evm' priority_vector method.
try:
    from sklearn.decomposition import PCA
except ImportError:
    PCA = None

from sklearn.linear_model import HuberRegressor

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


class _BootstrapWarnings:
    """
    Collect the warnings raised while calibrating bootstrap resamples and
    re-emit one aggregated warning per distinct message, instead of one per
    resample -- a 500-sample bootstrap can otherwise flood the user with
    hundreds of near-identical warnings (e.g. priority-vector
    non-convergence, or a connectivity warning that only affects a few
    unlucky resamples). Messages are grouped by category and by their text
    with numbers masked, since item names/counts and convergence figures
    vary between resamples; each group is counted at most once per
    resample, and its first message is shown as the example.

    Usage::

        collector = _BootstrapWarnings()
        for ...:
            with collector.resample():
                ...  # construct / calibrate one resample
        collector.emit()
    """

    _NUMBER = re.compile(r"\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")

    def __init__(self):
        self.no_of_resamples = 0
        self._groups = {}

    @contextmanager
    def resample(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            yield
        self.no_of_resamples += 1
        seen = set()
        for w in caught:
            key = (w.category, self._NUMBER.sub("#", str(w.message)))
            if key in seen:
                continue
            seen.add(key)
            if key in self._groups:
                self._groups[key][0] += 1
            else:
                self._groups[key] = [1, w.message]

    def emit(self, stacklevel=3):
        for (category, _), (count, message) in self._groups.items():
            if issubclass(category, (DeprecationWarning, PendingDeprecationWarning, FutureWarning)):
                # Library/API deprecations aren't about the resampled data,
                # so pass them through once, unchanged.
                warnings.warn(message, category, stacklevel=stacklevel)
                continue
            warnings.warn(
                f"{count} of {self.no_of_resamples} bootstrap resamples raised "
                f"the following warning (first occurrence shown; details such "
                f"as item names and numbers may differ between resamples):\n"
                f"{message}",
                category,
                stacklevel=stacklevel,
            )
        self._groups = {}


def _darken(color, lightness=0.35):
    """Darker shade of a colour (same hue/saturation, HLS lightness capped)."""
    h, l, sat = colorsys.rgb_to_hls(*colors.to_rgb(color))
    return colorsys.hls_to_rgb(h, min(l, lightness), sat)


def _save_table(table, filename, file_format="csv"):
    """
    Save a stats table returned by a stats method (item_stats_df etc.)
    when that method is given filename=. filename excludes the extension,
    matching the plotting methods' filename/file_format convention.
    """
    if filename is None:
        return
    if file_format == "csv":
        table.to_csv(f"{filename}.csv")
    elif file_format == "xlsx":
        table.to_excel(f"{filename}.xlsx")
    else:
        raise ValueError(f"file_format must be 'csv' or 'xlsx', not {file_format!r}.")


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

    # Named plot_style schemes shared by every plot_data()/wright_map()/
    # plot_anchor_selection()/std_residuals_hist() call across all four
    # model classes. 'base' is the seaborn style passed to sns.set_style()
    # (controls whether gridlines/spines are on by default); 'rc' is a
    # dict of rcParam overrides applied on top of it. 'white'/'dark' are
    # the original two schemes and carry no overrides, so they are
    # byte-for-byte unchanged from before this dict existed.
    _PLOT_STYLE_RC = {
        "white": {"base": "whitegrid", "rc": {}},
        "dark": {"base": "darkgrid", "rc": {}},
        "minimal": {"base": "white", "rc": {}},
        "black": {
            "base": "darkgrid",
            "rc": {
                "figure.facecolor": "#000000",
                "axes.facecolor": "#000000",
                "axes.edgecolor": "#4D4D4D",
                "axes.labelcolor": "#FFFFFF",
                "text.color": "#FFFFFF",
                "xtick.color": "#FFFFFF",
                "ytick.color": "#FFFFFF",
                "grid.color": "#333333",
            },
        },
        "parchment": {
            "base": "whitegrid",
            "rc": {
                "figure.facecolor": "#F4ECD8",
                "axes.facecolor": "#F4ECD8",
                "axes.edgecolor": "#8B7355",
                "axes.labelcolor": "#4A3728",
                "text.color": "#4A3728",
                "xtick.color": "#4A3728",
                "ytick.color": "#4A3728",
                "grid.color": "#D8C9A3",
            },
        },
        "print": {
            "base": "whitegrid",
            "rc": {
                "figure.facecolor": "#FFFFFF",
                "axes.facecolor": "#FFFFFF",
                "axes.edgecolor": "#000000",
                "axes.labelcolor": "#000000",
                "text.color": "#000000",
                "xtick.color": "#000000",
                "ytick.color": "#000000",
                "grid.color": "#000000",
                "grid.linewidth": 0.6,
                "axes.linewidth": 1.3,
            },
        },
        "solarized-light": {
            "base": "whitegrid",
            "rc": {
                "figure.facecolor": "#FDF6E3",
                "axes.facecolor": "#FDF6E3",
                "axes.edgecolor": "#93A1A1",
                "axes.labelcolor": "#586E75",
                "text.color": "#586E75",
                "xtick.color": "#586E75",
                "ytick.color": "#586E75",
                "grid.color": "#EEE8D5",
            },
        },
        "solarized-dark": {
            "base": "darkgrid",
            "rc": {
                "figure.facecolor": "#002B36",
                "axes.facecolor": "#002B36",
                "axes.edgecolor": "#586E75",
                "axes.labelcolor": "#93A1A1",
                "text.color": "#93A1A1",
                "xtick.color": "#93A1A1",
                "ytick.color": "#93A1A1",
                "grid.color": "#073642",
            },
        },
        "slate": {
            "base": "darkgrid",
            "rc": {
                "figure.facecolor": "#2E3440",
                "axes.facecolor": "#2E3440",
                "axes.edgecolor": "#8892A0",
                "axes.labelcolor": "#D8DEE9",
                "text.color": "#D8DEE9",
                "xtick.color": "#D8DEE9",
                "ytick.color": "#D8DEE9",
                "grid.color": "#3B4252",
            },
        },
        "blueprint": {
            "base": "darkgrid",
            "rc": {
                "figure.facecolor": "#0B3D91",
                "axes.facecolor": "#0B3D91",
                "axes.edgecolor": "#9FBFEE",
                "axes.labelcolor": "#E8F0FC",
                "text.color": "#E8F0FC",
                "xtick.color": "#E8F0FC",
                "ytick.color": "#E8F0FC",
                "grid.color": "#3E6BC4",
            },
        },
        "newsprint": {
            "base": "whitegrid",
            "rc": {
                "figure.facecolor": "#E7E7E2",
                "axes.facecolor": "#E7E7E2",
                "axes.edgecolor": "#2B2B2B",
                "axes.labelcolor": "#1A1A1A",
                "text.color": "#1A1A1A",
                "xtick.color": "#1A1A1A",
                "ytick.color": "#1A1A1A",
                "grid.color": "#B7B7AF",
            },
        },
        "chalkboard": {
            "base": "darkgrid",
            "rc": {
                "figure.facecolor": "#1B2E22",
                "axes.facecolor": "#1B2E22",
                "axes.edgecolor": "#D9E5D9",
                "axes.labelcolor": "#F1F1E6",
                "text.color": "#F1F1E6",
                "xtick.color": "#F1F1E6",
                "ytick.color": "#F1F1E6",
                "grid.color": "#3A5445",
            },
        },
    }

    # plot_style schemes whose background is dark/saturated enough that
    # the 'colorblind multi' palette's black entry (Okabe & Ito's own
    # 8th colour) would be invisible or near-invisible against it. Curve-
    # colouring code checks this set and substitutes white for that one
    # entry when the active plot_style is a member -- see palette_dict
    # construction in each model class's plot_data(). Deliberately
    # excludes 'dark' (seaborn darkgrid's light-grey background, against
    # which black remains perfectly legible).
    _DARK_BACKGROUND_STYLES = {
        "black", "solarized-dark", "slate", "blueprint", "chalkboard",
    }

    def _apply_plot_style(self, plot_style):
        """
        Apply a named plot_style scheme via seaborn/matplotlib rcParams.

        Shared by every plot_data()/wright_map()/plot_anchor_selection()/
        std_residuals_hist() call across all four model classes, so a new
        scheme only needs to be added once, here, to become available
        everywhere. Falls back to 'white' for an unrecognised name rather
        than raising, matching the original if/else's own silent
        fallback behaviour.

        Parameters
        ----------
        plot_style : str
            One of self._PLOT_STYLE_RC's keys: 'white', 'dark', 'black',
            'parchment', 'minimal', 'print', 'solarized-light',
            'solarized-dark', 'slate', 'blueprint', 'newsprint', or
            'chalkboard'.
        """
        scheme = self._PLOT_STYLE_RC.get(plot_style, self._PLOT_STYLE_RC["white"])
        sns.set_style(scheme["base"], scheme["rc"])

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

        Finally checks *strong* connectivity of the directed win-graph (an
        edge i -> j whenever someone scored higher on i than j). The
        Bradley-Terry maximum-likelihood priority vector
        (calibrate(method='log-lik')) has a finite, unique solution if and
        only if this digraph is strongly connected (Ford, 1957). When the
        graph is weakly but not strongly connected, a UserWarning names the
        strongly-connected components and points to the remedies:
        method='cos'/'ls'/'evm', or matrix_power >= 1 to densify first, or a
        larger smoothing constant. The other methods are unaffected — they
        run on the powered (matrix_power=5) matrix by default.

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
            - 'strongly_connected' : bool — True if the directed win-graph is
              strongly connected (required for a finite method='log-lik'
              solution on an unpowered matrix).
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
                "strongly_connected": False,
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
                "strongly_connected": False,
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

        # Rows/columns explained by an unused score category (PCM's
        # item-category grain): the unused-category warning already reports
        # the cause, and the default smoothing constant resolves it in
        # calibration, so the structural-zero and strong-connectivity
        # warnings below are skipped when these explain them fully.
        explained = np.asarray(self._unused_category_rows(row_items), dtype=bool)
        unexplained_zero = structurally_zero_rows & ~explained
        directionally_isolated_report = sorted(set(row_items[unexplained_zero].tolist()))
        if directionally_isolated_report:
            warnings.warn(
                f"{len(directionally_isolated_report)} item(s) have a structurally "
                f"unresolvable zero in calibrate()'s directed pairwise matrix: "
                f"{directionally_isolated_report}. This can silently corrupt "
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

        # Strong connectivity of the directed win-graph (edge i -> j whenever
        # someone scored higher on i than j), checked on the raw matrix at
        # its native grain. The Bradley-Terry MLE extracted by
        # calibrate(method='log-lik') is finite and unique only if this
        # digraph is strongly connected (Ford, 1957); the other priority
        # vector methods run on the powered matrix and are unaffected.
        directed = np.asarray(matrix) > 0
        np.fill_diagonal(directed, False)
        n_scc, scc_labels = connected_components(
            directed, directed=True, connection="strong"
        )
        strongly_connected = n_scc == 1
        scc_explained = False
        if not strongly_connected and explained.any():
            keep = ~explained
            n_rest, _ = connected_components(directed[np.ix_(keep, keep)], directed=True, connection="strong")
            scc_explained = n_rest == 1

        if is_connected:
            if not strongly_connected and not scc_explained:
                scc_groups = {}
                for node_idx, label in enumerate(scc_labels):
                    scc_groups.setdefault(label, set()).add(
                        str(row_items[node_idx])
                    )
                scc_groups = sorted(
                    (sorted(group) for group in scc_groups.values()), key=len
                )
                scc_summary = "; ".join(
                    f"Group {i + 1} (size {len(group)}): {group[:5]}"
                    for i, group in enumerate(scc_groups[:6])
                )
                if len(scc_groups) > 6:
                    scc_summary += f"; ... and {len(scc_groups) - 6} more"
                warnings.warn(
                    f"The directed comparison graph is connected but not "
                    f"*strongly* connected: it splits into {n_scc} "
                    f"strongly-connected components ({scc_summary}). "
                    f"calibrate(method='log-lik') runs on the raw, unpowered "
                    f"pairwise matrix by default and has no finite "
                    f"maximum-likelihood solution in this case (affected "
                    f"estimates diverge and hit the iteration cap). Use "
                    f"method='cos'/'ls'/'evm', pass matrix_power>=1 to densify "
                    f"first, or raise the smoothing constant. The other methods "
                    f"are unaffected.",
                    UserWarning,
                    stacklevel=2,
                )
            return {
                "connected": True,
                "components_count": 1,
                "directionally_isolated_items": directionally_isolated_items,
                "strongly_connected": bool(strongly_connected),
            }
        else:
            isolated_items = [comp for comp in components if len(comp) == 1]
            sub_group_summary = "; ".join(
                f"Sub-group {i + 1} (size {len(comp)}): {comp[:5]}"
                for i, comp in enumerate(components)
            )
            warnings.warn(
                f"The response data is disconnected into {len(components)} separate "
                f"sub-networks, with no empirical comparisons spanning them: "
                f"{sub_group_summary}. Items in different sub-networks cannot be "
                f"placed on a single scale -- each sub-network's estimates are "
                f"identified only up to its own origin -- and un-smoothed "
                f"calibration breaks down.",
                UserWarning,
                stacklevel=2,
            )
            return {
                "connected": False,
                "components_count": len(components),
                "isolated_items": isolated_items,
                "all_sub_groups": components,
                "directionally_isolated_items": directionally_isolated_items,
                "strongly_connected": bool(strongly_connected),
            }

    def _unused_category_rows(self, row_items):
        """Boolean mask over the pairwise matrix's rows marking rows whose
        structural zeros are explained by an unused score category (only
        PCM's item-category grain has such rows; overridden there)."""
        return np.zeros(len(row_items), dtype=bool)

    @staticmethod
    def _resolve_matrix_power(method, matrix_power):
        """
        Resolve the ``matrix_power=None`` sentinel used by the ``calibrate*``
        methods.

        ``None`` selects a method-dependent default: ``0`` (no powering) for
        ``method='log-lik'`` and ``5`` for every other method. The
        Bradley-Terry MM update consumed by ``'log-lik'`` handles an
        incomplete comparison graph natively (unobserved pairs simply drop
        out of the sum), and the entries of a powered matrix are transitive
        path counts rather than Bernoulli win counts, so powering only
        distorts its likelihood. ``'cos'``/``'ls'``/``'evm'`` instead need a
        densified matrix. The default was 3 (arbitrary); a power sweep on
        well-conditioned data found SLM and RSM fully plateaued by power 2-3
        (3 vs 5 bit-identical) while PCM's wider-diameter item-threshold
        graph keeps improving to ~power 5-6 (item RMSE -2%, recovery slope
        closer to 1), so 5 is now the default -- free for SLM/RSM, a small
        real gain for PCM, at the cost of two extra matrix multiplies on
        this non-default path.

        An explicit ``matrix_power=0`` is honoured for any method, but warns
        for the non-``'log-lik'`` methods since they assume no structural
        zeroes remain.
        """
        if matrix_power is None:
            return 0 if method == "log-lik" else 5
        if matrix_power == 0 and method != "log-lik":
            warnings.warn(
                f"matrix_power=0 leaves structural zeroes in the pairwise "
                f"matrix; method={method!r} assumes a densified matrix and "
                f"may give unstable estimates. Use method='log-lik', or "
                f"matrix_power>=1, or a non-zero smoothing constant.",
                UserWarning,
                stacklevel=3,
            )
        return matrix_power

    def priority_vector(
        self, matrix, method="log-lik", log_lik_tol=0.000001, pcm=False, raters=False
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
        method : str, default 'log-lik'
            Priority vector extraction method:
            'log-lik'  — Bradley-Terry maximum likelihood (default), via the
                         Hunter (2004) MM update warm-started from the 'cos'
                         solution: a fixed-point iteration on the (small)
                         pairwise matrix, not a data-scale procedure like
                         JML/MML. Handles an incomplete comparison graph
                         directly, so the calibrate* methods default it to an
                         unpowered matrix (matrix_power=None -> 0). Lowest
                         recovery RMSE of the four methods in simulation —
                         a trivial margin on dense data, up to ~10% under
                         sparsity. Needs a strongly connected win-graph (see
                         check_data_connectivity); falls over otherwise, where
                         the closed-form methods below still return an answer.
            'cos'      — cosine maximisation method (Kou & Lin, 2014): sum of
                         L2-normalised columns of the reciprocal matrix. Fast,
                         robust, closed-form; least sensitive to heavy-tailed
                         cell ratios, so no single column dominates.
            'ls'       — least squares (row mean of reciprocal matrix).
            'evm'      — eigenvector method (Garner & Engelhard, 2002) via PCA.
                         Requires scikit-learn.
        log_lik_tol : float, default 0.000001
            Convergence tolerance for the 'log-lik' method.
        pcm : bool, default False
            If True, names output using item-threshold labels for PCM calibration.
        raters : bool, default False
            If True, names output using facet element (e.g. rater) labels for MFRM calibration.

        Returns
        -------
        pandas.Series
            Item (or facet effect) estimates, zero-centred logits,
            indexed by item (or facet/rater) name. Returns None if 'evm' fails.
        """
        matrix_dim = matrix.shape[0]

        if pcm:
            # Cache the item-threshold labels: priority_vector() is called once
            # per bootstrap resample, and the nested f-string build is pure
            # overhead after the first call. Keyed on the column names so a
            # rename between calls rebuilds them.
            cols = tuple(self.responses.columns)
            if getattr(self, "_pcm_priority_names_key", None) != cols:
                self._pcm_priority_names_key = cols
                self._pcm_priority_names = [
                    f"{str(item)}_{str(j + 1)}"
                    for i, item in enumerate(cols)
                    for j in range(self.max_score_vector.iloc[i])
                ]
            names = self._pcm_priority_names
        else:
            names = self.facet_names if raters else list(self.responses.columns)

        with np.errstate(divide="ignore", invalid="ignore"):
            recip_matrix = np.divide(matrix.T, matrix)
            # posinf/neginf fills collapse structural zeroes to "equal
            # difficulty" (only reached on an unpowered/sparse matrix; the
            # default powered matrix has no zero cells).
            recip_matrix = np.nan_to_num(
                recip_matrix, nan=1.0, posinf=1.0, neginf=1.0, copy=False
            )
        # Force the reciprocal-matrix diagonal to the AHP convention a_ii = 1,
        # which 'cos', 'ls' and 'evm' (and the log-lik warm start) all assume.
        # Done explicitly rather than relying on the 0/0 -> nan -> 1.0 route, so
        # it still holds if the caller passes an unpowered matrix (matrix_power
        # = 0) whose diagonal is a real zero, or a powered matrix whose diagonal
        # is some positive path count.
        np.fill_diagonal(recip_matrix, 1.0)

        if method == "evm":
            if PCA is None:
                raise ImportError(
                    "scikit-learn is required for the 'evm' method. "
                    "Install it with: pip install scikit-learn"
                )
            # Only the first principal axis is used, so cap the decomposition.
            pca = PCA(n_components=1)
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
            # Bradley-Terry strengths via the Hunter (2004) MM update.
            # Warm-start from the closed-form cosine-maximisation solution,
            # inverted: 'cos' weights track exp(difficulty) whereas BT strengths
            # track exp(-difficulty). This starts near the optimum and typically
            # cuts the iteration count several-fold versus a wins-proportional
            # start.
            col_norms = np.linalg.norm(recip_matrix, axis=0)
            cos_weights = (recip_matrix / col_norms[np.newaxis, :]).sum(axis=1)
            weights = 1.0 / cos_weights
            weights /= weights.sum()

            # Wins and pairwise totals both exclude the diagonal: matrix powers
            # leave a non-zero diagonal, but i-vs-i is not a real comparison and
            # must drop out of both the numerator and denominator of the update.
            wins = matrix.sum(axis=1) - np.diagonal(matrix)
            matrix_sum_sym = matrix + matrix.T
            np.fill_diagonal(matrix_sum_sym, 0.0)

            weight_pairs = np.empty((matrix_dim, matrix_dim))
            term_matrix = np.empty((matrix_dim, matrix_dim))
            max_iter = 10000
            for _ in range(max_iter):
                np.maximum(weights, 1e-12, out=weights)
                np.add(
                    weights[:, np.newaxis], weights[np.newaxis, :], out=weight_pairs
                )
                np.divide(matrix_sum_sym, weight_pairs, out=term_matrix)
                adjustment = term_matrix.sum(axis=1)

                new_weights = wins / np.maximum(adjustment, 1e-12)
                new_weights /= new_weights.sum()

                change = np.max(np.abs(weights - new_weights))
                weights = new_weights
                if change <= log_lik_tol:
                    break
            else:
                warnings.warn(
                    f"'log-lik' priority vector did not converge within "
                    f"{max_iter} iterations (last change={change:.2e}).",
                    UserWarning,
                    stacklevel=2,
                )

            measures = -np.log(weights)
            measures -= np.mean(measures)
            measures = pd.Series(measures, index=names)

        else:
            if method == "ls":
                weights = np.mean(recip_matrix, axis=1)
            else:
                normaliser = np.linalg.norm(recip_matrix, axis=0)
                weights = (recip_matrix / normaliser[np.newaxis, :]).sum(axis=1)

            measures = np.log(weights)
            measures -= np.mean(measures)
            measures = pd.Series(measures, index=names)

        return measures

    # Threshold distances whose CPAT comparison patterns are observed fewer
    # than this many times in total (summed over ordered pairs of different
    # items) use the information-weighted estimator under 'cpat-cml'.
    CPAT_THIN_COUNT = 20

    @staticmethod
    def _cpat_info_distance(num, den, loc, constant):
        """
        Information-weighted CPAT threshold distance (Elliott & Buttery, 2022,
        eq. 4.18): the weighted mean over item pairs of
        log(num + c) - log(den + c) + (delta_i - delta_j), with harmonic-mean
        weights 2ab / (a + b). Pairs with no observations are dropped.

        num, den : arrays (I, I), diagonals zero -- counts of persons scoring
            (k, k) on items (i, j), and (k-1, k+1).
        loc : array (I, I) of item location differences delta_i - delta_j.
        constant : float, additive smoothing for observed pairs.

        Returns NaN if no pair carries any weight.
        """
        valid = (num + den) > 0
        a = np.where(valid, num + constant, 0.0)
        b = np.where(valid, den + constant, 0.0)
        finite = valid & (a > 0) & (b > 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_ratio = np.log(np.where(finite, a, 1.0)) - np.log(np.where(finite, b, 1.0))
            weight = np.where(finite, 2.0 * a * b / (a + b), 0.0)
        total = weight.sum()
        if total == 0:
            return np.nan
        return float((weight * (log_ratio + loc)).sum() / total)

    @staticmethod
    def _cpat_cml_distance(num, den, loc, constant):
        """
        Conditional maximum likelihood CPAT threshold distance.

        Given that a person's responses to items i != j are (k, k) or
        (k-1, k+1), theta cancels and
        logit P(k, k) = D - (delta_i - delta_j), with D = tau_{k+1} - tau_k,
        so num_ij ~ Binomial(num_ij + den_ij, p_ij) with the item differences
        as a known offset. D maximises the resulting binomial likelihood
        (strictly concave), found by Newton's method safeguarded by a bracket
        on the root of the score (which is strictly decreasing in D), with
        bisection whenever a step leaves the bracket. Unlike the
        information-weighted estimator, the weights are the model's own
        probabilities at D rather than the observed proportions, so they are
        not correlated with the sampling noise and do not compress the scale.

        Same arguments as _cpat_info_distance(). Returns NaN if no pair has
        any observations, or the smoothed counts are zero on either side.
        """
        mask = (num + den) > 0
        if not mask.any():
            return np.nan
        n = num[mask] + constant
        d = den[mask] + constant
        offset = loc[mask]
        total = n + d
        if n.sum() <= 0 or d.sum() <= 0:
            return np.nan

        def score(D):
            with np.errstate(over="ignore"):
                return (n - total / (1.0 + np.exp(-(D - offset)))).sum()

        with np.errstate(over="ignore", divide="ignore"):
            D = float(np.log(n.sum() / (d * np.exp(-offset)).sum()))
        if not np.isfinite(D):
            D = 0.0
        lo, hi = D - 1.0, D + 1.0
        while score(lo) < 0:
            lo -= 2.0 * (hi - lo)
        while score(hi) > 0:
            hi += 2.0 * (hi - lo)
        for _ in range(200):
            with np.errstate(over="ignore"):
                p = 1.0 / (1.0 + np.exp(-(D - offset)))
            g = (n - total * p).sum()
            h = (total * p * (1.0 - p)).sum()
            if g > 0:
                lo = D
            else:
                hi = D
            new = D + g / h if h > 0 else np.nan
            if not (lo < new < hi):
                new = 0.5 * (lo + hi)
            if abs(new - D) < 1e-10 or hi - lo < 1e-10:
                return float(new)
            D = new
        return float(D)

    @staticmethod
    def _origin_pinv(L):
        """
        Moore-Penrose inverse of a Bradley-Terry curvature matrix whose only
        null direction is the common origin (the vector of ones), computed
        exactly as (L + 11'/p)^-1 - 11'/p. np.linalg.pinv's tolerance can
        mistake that direction's rounding-level eigenvalue (~1e-12) for a real
        one and invert it, adding a spurious variance of order 1e12 along the
        origin.
        """
        p = L.shape[0]
        J = np.full((p, p), 1.0 / p)
        try:
            Lj = np.linalg.inv(L + J)
            if np.all(np.isfinite(Lj)) and np.linalg.cond(L + J) < 1e10:
                return Lj - J
        except np.linalg.LinAlgError:
            pass
        # More than one null direction (disconnected comparisons): drop every
        # eigenvalue below a relative tolerance well above rounding level
        vals, vecs = np.linalg.eigh((L + L.T) / 2)
        keep = vals > 1e-10 * np.abs(vals).max()
        return (vecs[:, keep] / vals[keep]) @ vecs[:, keep].T

    @staticmethod
    def _bt_person_curvature(pairs, v):
        """
        Per-person Bradley-Terry curvature matrices for a block of persons:
        pairs is a list of (R, C) arrays (persons x strengths), person n
        contributing a comparison between a and b for each R_na * C_nb and
        R_nb * C_na; v holds pi_ab (1 - pi_ab) (zero where not compared).
        Returns an (n, p, p) array whose sum over persons is the bread.
        """
        Nn = 0.0
        for R, C in pairs:
            Nn = Nn + R[:, :, None] * C[:, None, :] + C[:, :, None] * R[:, None, :]
        w = Nn * v[None]
        Ln = -w
        diag = np.arange(v.shape[0])
        Ln[:, diag, diag] = w.sum(axis=2)
        return Ln

    @staticmethod
    def _kc_scores(U, person_bread, bread_inv, chunk=500):
        """
        Kauermann-Carroll (2001) small-sample correction of a sandwich's
        per-person scores (the cluster form of Bell & McCaffrey, 2002): each
        person's score u_n becomes (I - H_n)^(-1/2) u_n, with leverage
        H_n = B_n B^+ (B_n the person's contribution to the bread B). The
        fitted scores are shrunk towards zero by (I - H_n) on average, since
        the estimates were chosen to make them sum to zero; the inverse
        square root removes that bias to first order. person_bread(sl)
        returns the bread contributions of persons U[sl] as an (n, p, p)
        array. A person whose I - H_n is near-singular keeps their score.
        """
        U = np.asarray(U, dtype=float)
        N, p = U.shape
        out = U.copy()
        eye = np.eye(p)
        for start in range(0, N, chunk):
            sl = slice(start, min(start + chunk, N))
            M = eye[None] - person_bread(sl) @ bread_inv
            w, Q = np.linalg.eig(M)
            w, Q = w.real, Q.real
            good = np.isfinite(w).all(axis=1) & (w.min(axis=1) > 1e-8)
            if not good.any():
                continue
            u = U[sl][good]
            try:
                y = np.linalg.solve(Q[good], u[:, :, None])[:, :, 0]
            except np.linalg.LinAlgError:
                continue
            adj = np.einsum("nij,nj->ni", Q[good], y / np.sqrt(w[good]))
            block = out[sl]
            block[good] = np.where(np.isfinite(adj), adj, u)
            out[sl] = block
        return out

    @staticmethod
    def _polytomous_moments(probs, cats, pf):
        """
        Expected score E, information I, third cumulant J and fourth
        cumulant K4 of the total score per person, from category
        probabilities.

        probs : array (K+1, N, I) of category probabilities.
        cats : array (K+1,) of category scores.
        pf : array (N, I), 1 where the item was observed, 0 or NaN where not.

        Items are independent given theta, so each cumulant of the total is
        the sum of the items' cumulants.
        """
        pf = np.nan_to_num(np.asarray(pf, dtype=float))
        mu = (cats[:, None, None] * probs).sum(axis=0)  # (N, I)
        dev = cats[:, None, None] - mu[None, :, :]
        dev2 = dev * dev
        var = (dev2 * probs).sum(axis=0)
        k3 = (dev2 * dev * probs).sum(axis=0)
        k4 = (dev2 * dev2 * probs).sum(axis=0) - 3.0 * var * var
        return (
            (mu * pf).sum(axis=1),
            (var * pf).sum(axis=1),
            (k3 * pf).sum(axis=1),
            (k4 * pf).sum(axis=1),
        )

    def _solve_wle(self, raw_scores, start, moments, tolerance=0.00001, max_iters=100):
        """
        Warm's (1989) weighted likelihood estimates, solved directly.

        Finds theta solving f(theta) = r - E(theta) + J(theta) / (2 I(theta)) = 0
        for each person, where r is the raw (unadjusted) score, E the expected
        score, I the information and J the third cumulant of the score. This
        replaces adding the one-step correction J / (2 I^2) to the ML
        estimate, which is only a first-order approximation: it is
        inaccurate where I is small and diverges for extreme scores (the
        smaller ext_score_adjustment, the more extreme the ML estimate and
        the larger the overshoot). The weighted likelihood equation has a
        finite root for every score, including 0 and the maximum, so no
        extreme-score adjustment is needed.

        Newton's method with the exact slope
        f'(theta) = -I + (K4 I - J^2) / (2 I^2), using dI/dtheta = J and
        dJ/dtheta = K4 (the fourth cumulant). f is positive below the root
        and negative above it, so every evaluation narrows a bracket around
        the root; steps are clipped to 1 logit until the root is bracketed,
        and a step leaving the bracket is replaced by bisection.

        Parameters
        ----------
        raw_scores : array-like (N,)
            Raw scores, without any extreme-score adjustment.
        start : array-like (N,)
            Starting values (normally the ML estimates). NaN entries are
            returned as NaN.
        moments : callable
            moments(theta, rows) -> (E, I, J, K4), each an array matching
            theta, for the persons at integer positions rows.
        tolerance : float, default 0.00001
            Convergence tolerance on theta.
        max_iters : int, default 100
            Maximum iterations.

        Returns
        -------
        numpy.ndarray
            WLE estimates; NaN where start was NaN or the solution did not
            converge.
        """
        r = np.asarray(raw_scores, dtype=float)
        theta = np.asarray(start, dtype=float).copy()
        rows = np.flatnonzero(np.isfinite(theta))
        th = theta[rows]
        lo = np.full(len(rows), -np.inf)
        hi = np.full(len(rows), np.inf)
        active = np.ones(len(rows), dtype=bool)

        for _ in range(max_iters):
            if not active.any():
                break
            a = np.flatnonzero(active)
            t = th[a]
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                e, i, j, k4 = moments(t, rows[a])
                f = r[rows[a]] - e + j / (2 * i)
                fp = -i + (k4 * i - j * j) / (2 * i * i)
                step = -f / fp
            finite = np.isfinite(f)
            pos = finite & (f > 0)
            neg = finite & (f <= 0)
            lo[a] = np.where(pos, t, lo[a])
            hi[a] = np.where(neg, t, hi[a])
            bracketed = np.isfinite(lo[a]) & np.isfinite(hi[a])
            # Newton step; uphill or undefined slope -> 1 logit towards the root
            bad = ~np.isfinite(step) | (fp >= 0)
            step = np.where(bad, np.where(pos, 1.0, -1.0), step)
            step = np.where(bracketed, step, np.clip(step, -1.0, 1.0))
            new = t + step
            outside = bracketed & ((new <= lo[a]) | (new >= hi[a]))
            new = np.where(outside, 0.5 * (lo[a] + hi[a]), new)
            # f undefined (information underflow far in a tail): back off
            new = np.where(finite, new, np.where(bracketed, 0.5 * (lo[a] + hi[a]), t - np.sign(t)))
            done = finite & (
                (np.abs(new - t) < tolerance) | (bracketed & (hi[a] - lo[a] < tolerance))
            )
            th[a] = new
            active[a[done]] = False

        out = np.full(theta.shape, np.nan)
        if active.any():
            warnings.warn(
                f"{int(active.sum())} person(s) did not converge in the Warm (WLE) "
                f"solution and will be set to NaN. Consider increasing max_iters.",
                UserWarning,
                stacklevel=3,
            )
            th[active] = np.nan
        out[rows] = th
        return out

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

            # Only re-apply style when caller explicitly requests a
            # non-default scheme, avoiding the per-call cost.
            if plot_style != "white":
                self._apply_plot_style(plot_style)

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

            plt.close(fig)

        return fig

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

    def _bootstrap_value_matrix(self, samples, attr, index):
        """
        (B, len(index)) pandas.DataFrame of a Series-valued attribute
        (e.g. items, facet_effects_global, lambda_centrality) across a
        model's own stored bootstrap resamples, columns=index. Samples
        whose attribute doesn't cover the full index (a resample that
        dropped a unit, or where calibration produced a ragged result)
        are skipped rather than filled with NaN, matching
        _store_rater_se's own validity filter.
        """
        rows = []
        for s in samples:
            val = getattr(s, attr, None)
            if val is None:
                continue
            val = val.reindex(index)
            if val.isna().any():
                continue
            rows.append(val.values)
        return pd.DataFrame(rows, columns=index) if len(rows) >= 2 else None

    def _bootstrap_element_matrices(self, samples, attr, index):
        """
        dict {element: (B, len(index)) pandas.DataFrame or None} for a
        DataFrame-valued attribute (unit x element, e.g.
        facet_effects_items) across a model's own stored bootstrap
        resamples -- the per-element analogue of _bootstrap_value_matrix.
        """
        first = next((getattr(s, attr, None) for s in samples), None)
        if first is None:
            return {}
        elements = first.columns
        collected = {el: [] for el in elements}
        for s in samples:
            val = getattr(s, attr, None)
            if val is None:
                continue
            val = val.reindex(index=index, columns=elements)
            for el in elements:
                col = val[el]
                if not col.isna().any():
                    collected[el].append(col.values)
        return {
            el: (pd.DataFrame(rows, columns=index) if len(rows) >= 2 else None)
            for el, rows in collected.items()
        }

    def _purification_loo_tc(
        self, ref_col, focal_col, ref_se_col, focal_se_col,
        selection_method, corr_tol, sd_ratio_tol, min_anchors, wald_alpha,
        ref_boot=None, focal_boot=None, ref_cov=None, focal_cov=None,
    ):
        """
        Leave-one-out translation constant for each tested unit, and the two
        independent variance components (reference group, focal group) of
        its purified difference. Shared by DIF (item-location purification,
        SLM/PCM/RSM/MFRM) and MFRM's DRF (facet element location/lambda
        purification).

        For each unit r, the anchor set is selected without r, and tc is
        computed from that set. The variance of
            diff = focal_col[r] + tc - ref_col[r]
                 = [focal_col[r] - anchor_mean(focal, sel)]
                   - [ref_col[r]  - anchor_mean(ref,  sel)]
        splits into two independent parts, one per group, each estimated:
        - from that group's covariance matrix (ref_cov/focal_cov, sandwich
          standard errors) when given: Var(x_r - a'x_sel) =
          V_rr - 2 a'V_sel,r + a'V_sel,sel a, with a the (weighted)
          anchor-mean coefficients;
        - otherwise from that group's stored bootstrap resamples
          (ref_boot/focal_boot), as the variance of the unit's deviation
          from the anchor mean across resamples;
        - otherwise from the independence formula sum(se_i^2)/n_anchors^2.

        Returns
        -------
        tc_per_unit : pandas.Series, indexed like ref_col -- computed
            WITHOUT r's own (ref_col[r], focal_col[r]) pair.
        ref_var_per_unit, focal_var_per_unit : pandas.Series --
            Var(ref_col[r] - anchor_mean(ref, sel)) and
            Var(focal_col[r] - anchor_mean(focal, sel)) respectively;
            se = sqrt(ref_var_per_unit + focal_var_per_unit).
        selected, anchor_table : the whole-set (not leave-one-out) anchor
            selection, for reporting only; not used for the values above.
        """
        idx = ref_col.index
        tc_per_unit = pd.Series(index=idx, dtype=float)
        ref_var_per_unit = pd.Series(index=idx, dtype=float)
        focal_var_per_unit = pd.Series(index=idx, dtype=float)
        for r in idx:
            others = idx.drop(r)
            ref_o, focal_o = ref_col.loc[others], focal_col.loc[others]
            ref_se_o, focal_se_o = ref_se_col.loc[others], focal_se_col.loc[others]
            w = None
            if selection_method == "wald":
                ar = self._wald_anchor_selection(
                    ref_o, focal_o, focal_se_o, alpha=wald_alpha, min_anchors=min_anchors,
                )
                sel = ar["selected_anchors"]
                tc_per_unit[r] = ar["tc"]
                w = 1.0 / focal_se_o.loc[sel] ** 2
            elif selection_method == "robust_z":
                ar = self._robust_anchor_selection(
                    ref_o, focal_o, corr_tol=corr_tol, sd_ratio_tol=sd_ratio_tol,
                    min_anchors=min_anchors,
                )
                sel = ar["selected_anchors"]
                tc_per_unit[r] = ar["tc"]
            else:
                sel = others
                tc_per_unit[r] = ref_o.mean() - focal_o.mean()
            n_sel = len(sel)

            have_boot = (
                ref_boot is not None and focal_boot is not None and n_sel > 0
                and set(sel).issubset(ref_boot.columns)
                and set(sel).issubset(focal_boot.columns)
                and r in ref_boot.columns and r in focal_boot.columns
            )
            if have_boot:
                if w is not None:
                    wn = (w / w.sum()).reindex(sel).values
                    ref_anchor_mean_b = ref_boot[sel].values @ wn
                    focal_anchor_mean_b = focal_boot[sel].values @ wn
                else:
                    ref_anchor_mean_b = ref_boot[sel].values.mean(axis=1)
                    focal_anchor_mean_b = focal_boot[sel].values.mean(axis=1)
                d_ref_b = ref_boot[r].values - ref_anchor_mean_b
                d_focal_b = focal_boot[r].values - focal_anchor_mean_b
                ref_var_per_unit[r] = float(np.var(d_ref_b, ddof=1))
                focal_var_per_unit[r] = float(np.var(d_focal_b, ddof=1))
            elif (
                ref_cov is not None and focal_cov is not None and n_sel > 0
                and set(sel).issubset(ref_cov.index) and set(sel).issubset(focal_cov.index)
                and r in ref_cov.index and r in focal_cov.index
            ):
                a = (w / w.sum()).reindex(sel).values if w is not None else np.full(n_sel, 1.0 / n_sel)
                for cov, store in ((ref_cov, ref_var_per_unit), (focal_cov, focal_var_per_unit)):
                    Vss = cov.loc[sel, sel].values
                    Vsr = cov.loc[sel, r].values
                    store[r] = float(cov.loc[r, r] - 2.0 * a @ Vsr + a @ Vss @ a)
            elif n_sel > 0:
                tc_var = float((ref_se_o.loc[sel] ** 2 + focal_se_o.loc[sel] ** 2).sum()) / (n_sel ** 2)
                ref_var_per_unit[r] = ref_se_col.loc[r] ** 2
                focal_var_per_unit[r] = focal_se_col.loc[r] ** 2 + tc_var
            else:
                ref_var_per_unit[r] = np.nan
                focal_var_per_unit[r] = np.nan

        if selection_method == "wald":
            ar_full = self._wald_anchor_selection(
                ref_col, focal_col, focal_se_col, alpha=wald_alpha, min_anchors=min_anchors,
            )
            selected, anchor_table = ar_full["selected_anchors"], ar_full["table"]
        elif selection_method == "robust_z":
            ar_full = self._robust_anchor_selection(
                ref_col, focal_col, corr_tol=corr_tol, sd_ratio_tol=sd_ratio_tol,
                min_anchors=min_anchors,
            )
            selected, anchor_table = ar_full["selected_anchors"], ar_full["table"]
        else:
            selected, anchor_table = idx, None

        return tc_per_unit, ref_var_per_unit, focal_var_per_unit, selected, anchor_table

    def _calibrate_anchor_core(
        self, anchors, selection_method, corr_tol, sd_ratio_tol, min_anchors,
        wald_alpha, no_of_samples, adj, overwrite_anchors, plot, plot_kwargs,
        seed, std_errors_kwargs,
    ):
        """
        Shared core of calibrate_anchor() across SLM/PCM/RSM.

        The subclasses pass their own calibrate()/std_errors() keyword
        arguments; each keeps its own threshold-compatibility test. MFRM
        has its own calibrate_anchor().

        Computes the translation constant (via adj / wald / robust_z /
        none selection), applies it to get anchor_items, resolves
        overwrite_anchors, builds anchor_summary, and (if applicable)
        triggers plot_anchor_selection() -- exactly the behaviour each
        subclass's own calibrate_anchor() docstring documents under
        the same parameter names, this is a pure implementation-sharing
        extraction with no behaviour change.

        Callers must already have anchors as a pd.Series and self.items
        already calibrated (calibrate()'s own re-run trigger differs
        slightly per subclass -- constant/robust/threshold_constant/etc.
        -- so that step stays in each subclass's own calibrate_anchor(),
        before calling this).

        Parameters
        ----------
        std_errors_kwargs : dict
            Forwarded to self.std_errors(no_of_samples=no_of_samples,
            seed=seed, **std_errors_kwargs) if selection_method='wald'
            needs to auto-trigger it (self.item_se not already present).
            Carries whichever calibration kwarg name this subclass's own
            calibrate()/std_errors() uses (e.g. {'constant':..., 'robust':
            ..., 'method':..., 'matrix_power':..., 'log_lik_tol':...} for
            SLM/PCM, or the RSM equivalent with 'threshold_constant'
            instead of 'robust') -- exactly what each subclass's own
            calibrate_anchor() forwarded inline before this extraction.

        See any of SLM.calibrate_anchor / PCM.calibrate_anchor /
        RSM.calibrate_anchor for the full parameter documentation (adj,
        overwrite_anchors, plot, plot_kwargs, etc.) -- identical meanings
        here, just centralised.

        Attributes set
        --------------
        anchor_selection, anchor_selected_items, anchor_dropped_items,
        anchor_original_corr, anchor_original_sd_ratio, anchor_corr,
        anchor_sd_ratio, anchor_adj, anchor_item_names, anchor_items,
        anchor_summary, anchor_plot -- see the subclass docstrings.
        """
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
                    no_of_samples=no_of_samples, seed=seed, **std_errors_kwargs
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

    def _snapshot_calibrate_anchor_run(self, anchors, extra=None):
        """
        Snapshot a calibrate_anchor() run into self.calibrate_anchor_runs,
        keyed by the anchors supplied, so results from an earlier call
        survive a later call with a different anchor set instead of being
        overwritten in place. Shared tail of SLM/PCM/RSM's
        calibrate_anchor().

        Parameters
        ----------
        anchors : pandas.Series
            The same anchors passed to _calibrate_anchor_core() -- used
            only to build the dict key (tuple(sorted(anchors.items()))).
        extra : dict or None, default None
            Subclass-specific extra attributes to include in the snapshot
            beyond the common set (e.g. PCM's anchor_threshold_test /
            anchor_threshold_test_aggregate, RSM's anchor_threshold_test /
            anchor_category_width_test). None for subclasses with no
            model-specific threshold test (SLM).
        """
        from types import SimpleNamespace

        if not hasattr(self, "calibrate_anchor_runs"):
            self.calibrate_anchor_runs = {}
        key = tuple(sorted(anchors.items()))
        fields = dict(
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
        if extra:
            fields.update(extra)
        self.calibrate_anchor_runs[key] = SimpleNamespace(**fields)

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

    def _dif_group_setup(self, non_extreme, covariate, reference):
        """
        Shared covariate/reference-focal group setup for dif_test(),
        identical across SLM/PCM/RSM/MFRM -- the only thing that differs
        between classes is how `non_extreme` itself is computed (a plain
        per-person score/max-score comparison for SLM/PCM/RSM, an
        aggregated-across-facet_elements version for MFRM), which the
        caller computes and passes in.

        Parameters
        ----------
        non_extreme : pandas.Index
            Non-extreme person names/index, computed by the caller.
        covariate : str
            Column name in self.exogenous to group persons by.
        reference : value or None
            Value of `covariate` to use as the reference group. If None,
            the largest group is used.

        Returns
        -------
        cov_values : pandas.Series
            Non-null covariate values for non-extreme persons, indexed
            by person.
        reference : the resolved reference-group value.
        focal_levels : list
            Covariate values to compare against the reference, in
            descending group-size order.
        group_sizes : dict {value: int}
            Non-extreme, non-missing-covariate N used for each group.
        n_missing : int
            Non-extreme persons excluded for having a missing covariate
            value.
        """
        if getattr(self, "exogenous", None) is None:
            raise ValueError(
                "No exogenous data available. Pass exogenous= to the "
                "constructor before calling dif_test()."
            )
        if covariate not in self.exogenous.columns:
            raise ValueError(f"'{covariate}' is not a column in self.exogenous.")

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
                stacklevel=3,
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
                stacklevel=3,
            )

        return cov_values, reference, focal_levels, group_sizes, n_missing

    def _dif_category_column(
        self, diff, se, p, common_items, welch, welch_df,
        category_thresholds, category_alpha,
    ):
        """
        ETS-style DIF Category column (Zwick, Thayer & Lewis, 1999),
        identical across SLM/PCM/RSM/MFRM's dif_test — a pure function of
        each item's purified Difference/SE/p (and Welch df, if welch=True),
        with no model-specific inputs.

        Returns
        -------
        pandas.Series of {'A', 'B+', 'B-', 'C+', 'C-'}, indexed like
        common_items.
        """
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
        return pd.Series(category_col, index=common_items)

    def _dif_lr_table(self, lr_rows, common_items, correction, alpha, diff, logit_threshold):
        """
        Post-processing shared by every class's per-item DIF LR test
        (test='lr'/'both'): multiple-comparison correction and the
        Flagged_LR gate. The LR statistics themselves (lr_rows, a dict
        {item: {'LR':..., 'p_LR':...}}) are computed by the caller, since
        that part IS genuinely model-specific (each class's own scratch-
        model reconstruction for H0).

        Returns
        -------
        pandas.DataFrame, index=common_items, columns 'LR', 'p_LR',
        'p_LR (corrected)', 'Flagged_LR'.
        """
        lr_table = pd.DataFrame(lr_rows).T.loc[common_items]
        if correction == "bonferroni":
            lr_table["p_LR (corrected)"] = (lr_table["p_LR"] * len(lr_table)).clip(upper=1.0)
        elif correction == "bh":
            lr_table["p_LR (corrected)"] = self._bh_correction(lr_table["p_LR"])
        else:
            lr_table["p_LR (corrected)"] = lr_table["p_LR"]
        lr_table["Flagged_LR"] = (lr_table["p_LR (corrected)"] < alpha) & (
            diff.abs() >= logit_threshold
        )
        return lr_table

    def _dif_validate(
        self, correction, selection_method, test, category_thresholds, reference_n,
        dcf_cap=None,
    ):
        """
        Shared dif_test() argument validation -- identical across SLM/PCM/
        RSM/MFRM aside from dcf_cap, which only applies to the three
        classes with a threshold-structure DIF/DCF component (SLM is
        dichotomous, no thresholds at all). Pass dcf_cap=None (the
        default) to skip that check.
        """
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
        if dcf_cap is not None and dcf_cap <= 0:
            raise ValueError("dcf_cap must be greater than 0.")

    def _dif_item_wald_block(
        self, ref_model, focal_model, ref_boot, focal_boot,
        selection_method, corr_tol, sd_ratio_tol, min_anchors, wald_alpha,
        size_adjust, reference_n, welch, correction, alpha, logit_threshold,
        category, category_thresholds, category_alpha,
        ref_cov=None, focal_cov=None,
    ):
        """
        Shared item-location purification + per-item Wald test block from
        dif_test() -- byte-identical across SLM/PCM/RSM/MFRM aside from
        which bootstrap matrix the caller supplies for tc's own variance
        (item_bootstrap for SLM/RSM, central_bootstrap for PCM, a matrix
        MFRM builds per-model via _bootstrap_value_matrix before calling
        this, since MFRM stores its bootstrap resamples per facet-element-
        parameterisation rather than under one fixed attribute name).
        Only the item-location component; each class's own threshold-
        structure DIF/DCF block (PCM's per-item, RSM/MFRM's shared-vector
        category-width test) is NOT part of this -- those differ by
        design (see calibrate_anchor's own docstring on why PCM/RSM's
        threshold handling isn't the same statistical procedure) and stay
        in each subclass.

        Returns
        -------
        dict with keys: common_items, tc, focal_shifted, diff, se, z, p,
        p_corrected, flagged, selected, anchor_table, welch_df
        (pandas.Series or None), category_col (pandas.Series or None).
        """
        common_items = ref_model.items.index.intersection(focal_model.items.index)
        ref_items = ref_model.items.loc[common_items]
        focal_items = focal_model.items.loc[common_items]
        ref_se_item = ref_model.item_se.loc[common_items]
        focal_se_item = focal_model.item_se.loc[common_items]

        # Leave-one-out per item, with tc's own variance -- see
        # _purification_loo_tc.
        tc, ref_var, focal_var, selected, anchor_table = self._purification_loo_tc(
            ref_items, focal_items, ref_se_item, focal_se_item,
            selection_method, corr_tol, sd_ratio_tol, min_anchors, wald_alpha,
            ref_boot=ref_boot, focal_boot=focal_boot, ref_cov=ref_cov, focal_cov=focal_cov,
        )

        focal_shifted = focal_items + tc
        diff = focal_shifted - ref_items

        ref_n_item = ref_model.no_of_persons
        focal_n_item = focal_model.no_of_persons
        if size_adjust:
            # Scales the purification-corrected VARIANCE by n/reference_n
            # -- equivalent to the old SE*sqrt(n/reference_n) scaling,
            # applied post-hoc so it only touches the final reported SE
            # (size_adjust never feeds back into the anchor-selection
            # weighting, still computed from the raw, unscaled item_se
            # above).
            ref_var = ref_var * (ref_n_item / reference_n)
            focal_var = focal_var * (focal_n_item / reference_n)
            ref_n_item = focal_n_item = reference_n

        ref_se_eff = np.sqrt(ref_var)
        focal_se_eff = np.sqrt(focal_var)
        se = np.sqrt(ref_var + focal_var)
        welch_df = None
        if welch:
            z_vals, df_vals, p_vals = self._welch_satterthwaite(
                diff.values, ref_se_eff.values, ref_n_item,
                focal_se_eff.values, focal_n_item,
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

        category_col = None
        if category:
            category_col = self._dif_category_column(
                diff, se, p, common_items, welch, welch_df,
                category_thresholds, category_alpha,
            )

        return {
            "common_items": common_items, "tc": tc, "focal_shifted": focal_shifted,
            "diff": diff, "se": se, "z": z, "p": p, "p_corrected": p_corrected,
            "flagged": flagged, "selected": selected, "anchor_table": anchor_table,
            "welch_df": welch_df, "category_col": category_col,
        }

    def _dif_item_table(
        self, focal, ref_model, focal_model, common_items, focal_shifted,
        diff, se, z, p, p_corrected, selected, flagged,
        welch_df=None, category_col=None, lr_table=None,
    ):
        """
        Shared item_table construction from dif_test() -- zero diff across
        SLM/PCM/RSM/MFRM once fed the block above's outputs.
        """
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
        if welch_df is not None:
            table["df"] = welch_df
        if category_col is not None:
            table["Category"] = category_col
        if lr_table is not None:
            table = table.join(lr_table)
        return table

    def _parameter_recovery_table(
        self, generating, estimated, se=None, alpha=0.05, correction="bh",
        groups=None, divergence_tol=0.1,
    ):
        """
        Build the per-parameter recovery table and summary table for one
        parameter type, comparing generating (true, or -- for persons --
        pseudo-estimate) values against estimated values.

        Shared by every model class's recovery_stats(). See recovery_stats()
        docstrings for the overall design (no-op on model mismatch, the
        person pseudo-estimate construction, anchor=/groups= semantics).

        Parameters
        ----------
        generating : pandas.Series
            True (or pseudo-estimate, for persons) values, indexed by
            parameter name.
        estimated : pandas.Series
            Recovered estimates. Aligned to generating by index via plain
            pandas subtraction -- a label present on one side only becomes
            a NaN row in the table (e.g. a person dropped as extreme, or an
            item excluded by connectivity checks), not an error.
        se : pandas.Series or None, default None
            Standard errors matching estimated's values. None disables the
            z/p/Flagged columns (left all-NaN/False) -- used for parameter
            types with no available SE.
        alpha : float, default 0.05
            Significance threshold for Flagged/Flagged_corrected.
        correction : {'bh', 'bonferroni', None}, default 'bh'
            Multiple-comparison correction for p_corrected, matching
            dif_test's own convention (self._bh_correction for BH; the
            package's standard inline (p*n).clip(upper=1) idiom for
            Bonferroni). None skips correction (p_corrected/Flagged_corrected
            left all-NaN/False).
        groups : pandas.Series or None, default None
            Per-parameter group membership (person-level grouping in
            practice -- see recovery_stats()'s groups= parameter). Adds a
            'Group' column to the table and one summary column per group
            (plus 'Overall', which always reflects every row regardless of
            group).
        divergence_tol : float, default 0.1
            Threshold for |OLS slope - Huber slope| beyond which
            Divergence_flag is set True in the summary.

        Returns
        -------
        table : pandas.DataFrame
            One row per parameter (union of generating/estimated's
            indices), columns: generating, estimated, SE, error, abs_error,
            z, p, p_corrected, Flagged, Flagged_corrected, and Group if
            groups was given.
        summary : pandas.DataFrame
            One column 'Overall' (plus one per distinct group value, sorted,
            if groups was given). Rows: RMSE, MAE, bias, r, r_squared,
            SD_ratio, OLS_slope, OLS_intercept, Huber_slope, Huber_intercept,
            Divergence_flag, Prop_flagged.
        """
        idx = generating.index.union(estimated.index)
        generating = generating.astype(float).reindex(idx)
        estimated = estimated.astype(float).reindex(idx)
        error = estimated - generating
        abs_error = error.abs()

        if se is not None:
            se = se.astype(float).reindex(idx)
            z = error / se
            p = pd.Series(2 * (1 - norm.cdf(z.abs())), index=idx)
        else:
            z = pd.Series(np.nan, index=idx)
            p = pd.Series(np.nan, index=idx)

        if correction == "bh":
            valid_p = p.dropna()
            p_corrected = self._bh_correction(valid_p).reindex(idx) if len(valid_p) else p.copy()
        elif correction == "bonferroni":
            n_valid = p.notna().sum()
            p_corrected = (p * n_valid).clip(upper=1.0)
        else:
            p_corrected = pd.Series(np.nan, index=idx)

        flagged = p < alpha
        flagged_corrected = p_corrected < alpha

        table = pd.DataFrame({
            "generating": generating,
            "estimated": estimated,
            "SE": se if se is not None else pd.Series(np.nan, index=idx),
            "error": error,
            "abs_error": abs_error,
            "z": z,
            "p": p,
            "p_corrected": p_corrected,
            "Flagged": flagged,
            "Flagged_corrected": flagged_corrected,
        })
        if groups is not None:
            table["Group"] = groups.reindex(idx)

        def _summary_slice(sub):
            valid = sub.dropna(subset=["generating", "estimated"])
            n = len(valid)
            g = valid["generating"].values
            e = valid["estimated"].values
            err = valid["error"].values
            rmse = float(np.sqrt(np.mean(err**2))) if n else np.nan
            mae = float(np.mean(np.abs(err))) if n else np.nan
            bias = float(np.mean(err)) if n else np.nan
            flagged_valid = sub["Flagged"].dropna()
            prop_flagged = float(flagged_valid.mean()) if len(flagged_valid) else np.nan

            if n < 3 or np.std(g) < 0.05:
                r = r_squared = ols_slope = ols_intercept = np.nan
                huber_slope = huber_intercept = sd_ratio = np.nan
                divergence_flag = np.nan
            else:
                r = float(np.corrcoef(g, e)[0, 1])
                r_squared = r ** 2
                sd_ratio = float(np.std(e) / np.std(g))
                ols_slope, ols_intercept = (float(v) for v in np.polyfit(g, e, 1))
                hr = HuberRegressor(epsilon=1.5, max_iter=200).fit(
                    g.reshape(-1, 1), e
                )
                huber_slope = float(hr.coef_[0])
                huber_intercept = float(hr.intercept_)
                divergence_flag = bool(
                    abs(ols_slope - huber_slope) > divergence_tol
                )

            return pd.Series({
                "RMSE": rmse, "MAE": mae, "bias": bias, "r": r,
                "r_squared": r_squared,
                "SD_ratio": sd_ratio, "OLS_slope": ols_slope,
                "OLS_intercept": ols_intercept, "Huber_slope": huber_slope,
                "Huber_intercept": huber_intercept,
                "Divergence_flag": divergence_flag,
                "Prop_flagged": prop_flagged,
            })

        summary = {"Overall": _summary_slice(table)}
        if groups is not None:
            for g_val in sorted(table["Group"].dropna().unique(), key=str):
                summary[str(g_val)] = _summary_slice(table[table["Group"] == g_val])
        summary = pd.DataFrame(summary)

        return table, summary

    def _parameter_estimation_residual(
        self, true_theta, exp_score_generating, exp_score_estimated,
    ):
        """
        RMS residual between expected scores under two item/threshold
        parameter sets, holding person location fixed at the TRUE
        generating value throughout (Elliott & Buttery, 2022).

        Unlike a raw parameter-scale residual (RMSE/error in
        _parameter_recovery_table), this is evaluated on the response-
        probability scale, which sidesteps any residual origin/scale
        indeterminacy between two independently-derived parameter sets --
        particularly relevant when the "estimated" side comes from a
        different package/algorithm with its own identification
        convention. Holding theta at the true value (not the recovered or
        pseudo-estimated one) isolates item/threshold recovery from
        person-side estimation noise, matching how the persons recovery
        table itself isolates item/threshold noise from response-pattern
        noise via a pseudo-estimate -- both constructions difference out
        one source of noise to isolate another.

        Only evaluated over self.responses' own observed (non-null)
        cells, so missing-data conditions are handled automatically.

        Parameters
        ----------
        true_theta : pandas.Series
            TRUE generating person locations, indexed by person -- NOT
            this object's own recovered or pseudo-estimated persons.
        exp_score_generating, exp_score_estimated : callable
            (theta, item) -> expected score, using the generating and
            estimated item/threshold parameters respectively.

        Returns
        -------
        float
            RMS residual, in expected-score units (same scale as the
            response data, e.g. [0, max_score]). NaN if no observed cells
            have a person present in true_theta.
        """
        observed = self.responses.notna()
        # MFRM's self.responses is indexed by (rater, person) tuples (a
        # facet-element model's row grid, not a plain person index like
        # SLM/PCM/RSM) -- person is always the innermost/last level of
        # that MultiIndex, so extract it there; a plain Index (the other
        # three models) is used as-is.
        is_multi = isinstance(observed.index, pd.MultiIndex)
        gens = []
        ests = []
        for item in observed.columns:
            row_keys = observed.index[observed[item]]
            for row_key in row_keys:
                person = row_key[-1] if is_multi else row_key
                if person not in true_theta.index:
                    continue
                theta = float(true_theta.loc[person])
                gens.append(exp_score_generating(theta, item))
                ests.append(exp_score_estimated(theta, item))
        if not gens:
            return float("nan")
        g = np.asarray(gens, dtype=float)
        e = np.asarray(ests, dtype=float)
        return float(np.sqrt(np.mean((e - g) ** 2)))

    def _recovery_plot(
        self, generating, estimated, label, filename=None, file_format="png",
        dpi=300, plot_style="white", font="Times New Roman",
        title_font_size=15, axis_font_size=12, labelsize=12, figsize=(8, 6),
        colors=None, markers=None,
    ):
        """
        Scatter plot of estimated vs. generating parameter values, with an
        identity line plus OLS (dashed) and Huber (dotted) fit lines.

        Parameters
        ----------
        generating, estimated : pandas.Series or dict of {label: Series}
            Plain Series for a single-group plot. A dict of
            {group_label: Series} (both generating and estimated, same
            keys) plots every group on ONE shared axes in its own colour
            with its own fit lines -- used for MFRM's bivector/bistretch
            sub-vectors, the by-facet-element comparison plot, and the anchor-vs-
            non-anchor facet element (e.g. rater) split (see MFRM.recovery_stats()'s anchor=
            handling).
        label : str
            Parameter-type label, used for axis titles ('Generating
            {label}' / 'Estimated {label}') and the plot title.
        colors, markers : dict of {group_label: value} or None
            Optional per-group overrides -- colors default to a cycled
            'tab10' palette, markers default to 'o', when not given for a
            particular group label (or at all).
        filename, file_format, dpi : save-to-file trio, as elsewhere in the
            package (plot_anchor_selection, wright_map, plot_data). Since
            recovery_stats() calls this once per parameter type sharing
            one filename= via **plot_kwargs, the saved file is actually
            f"{filename}_{label}.{file_format}" (label lowercased, spaces
            and parentheses stripped) -- each parameter type gets its own
            file rather than overwriting the last one.
        plot_style, font, title_font_size, axis_font_size, labelsize, figsize :
            Same kwarg vocabulary as wright_map/plot_data/plot_anchor_selection.

        Returns
        -------
        matplotlib.figure.Figure
        """
        self._apply_plot_style(plot_style)
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_aspect("equal")

        if isinstance(generating, dict):
            groups_dict = {k: (generating[k], estimated[k]) for k in generating}
        else:
            groups_dict = {label: (generating, estimated)}

        colors = colors or {}
        markers = markers or {}
        cmap = plt.get_cmap("tab10")
        all_vals = []
        for i, (glabel, (g, e)) in enumerate(groups_dict.items()):
            g = np.asarray(g, dtype=float)
            e = np.asarray(e, dtype=float)
            mask = np.isfinite(g) & np.isfinite(e)
            g, e = g[mask], e[mask]
            if len(g) == 0:
                continue
            color = colors.get(glabel, cmap(i % 10))
            marker = markers.get(glabel, "o")
            ax.scatter(
                g, e, alpha=0.6, s=30, color=color, marker=marker,
                edgecolors="k", linewidths=0.3, label=str(glabel), zorder=3,
            )
            all_vals.append(g)
            all_vals.append(e)
            if len(g) >= 2:
                m, b = np.polyfit(g, e, 1)
                xseq = np.linspace(g.min(), g.max(), 50)
                ax.plot(xseq, m * xseq + b, color=color, linestyle="--", lw=1.2)
            if len(g) >= 3:
                hr = HuberRegressor(epsilon=1.5, max_iter=200).fit(
                    g.reshape(-1, 1), e
                )
                xseq = np.linspace(g.min(), g.max(), 50)
                ax.plot(
                    xseq, hr.coef_[0] * xseq + hr.intercept_, color=color,
                    linestyle=":", lw=1.2,
                )

        if all_vals:
            combined = np.concatenate(all_vals)
            lo, hi = float(combined.min()), float(combined.max())
            xseq = np.linspace(lo, hi, 100)
            ax.plot(xseq, xseq, color="black", lw=1, label="Identity", zorder=2)

        ax.set_xlabel(
            f"Generating {label}", fontsize=axis_font_size, fontweight="bold",
            fontname=font,
        )
        ax.set_ylabel(
            f"Estimated {label}", fontsize=axis_font_size, fontweight="bold",
            fontname=font,
        )
        ax.set_title(
            f"{label} recovery", fontsize=title_font_size, fontweight="bold",
            fontname=font,
        )
        ax.tick_params(axis="x", labelsize=labelsize)
        ax.tick_params(axis="y", labelsize=labelsize)
        ax.legend()

        if filename is not None:
            # recovery_stats() calls this once per parameter type, all
            # sharing the same filename= via **plot_kwargs -- suffix with
            # the (sanitised) label so each plot gets its own file instead
            # of every call silently overwriting the last one.
            suffix = label.lower().replace(" ", "_").replace("(", "").replace(")", "")
            fig.savefig(f"{filename}_{suffix}.{file_format}", dpi=dpi)

        plt.show(block=False)
        plt.pause(0.001)

        return fig

    def _pseudo_estimate_persons(self):
        """
        Re-run Newton-Raphson person estimation on self.responses using the
        TRUE generating item/threshold parameters, for the Elliott &
        Buttery (2022) person-recovery construction shared by every model
        class's recovery_stats().

        Dispatches on self.generating.model (SLM/PCM/RSM), NOT on
        self.model/type(self) -- person recovery is valid across any
        appropriate pairing (e.g. an RSM object's persons compared against
        pseudo-estimates built from PCM_Sim-generated thresholds, or vice
        versa), unlike item/threshold recovery which needs the two
        structures to be genuinely nested to mean anything (see each
        model's recovery_stats() docstring). MFRM's own version of this
        (which also needs to reconstruct a full facet-effect grid)
        lives in mfrm.py instead, since it needs MFRM-specific machinery
        (_sev_attr, per-submodel reconstruction) this shared version
        doesn't need for SLM/PCM/RSM.

        Uses a scratch/probe instance that never mutates self, matching
        the convention already used by RSM._threshold_structure_ll/
        std_errors/andersen_lr_test.

        Returns
        -------
        pandas.Series
            Pseudo-estimate person locations, indexed like self.persons.
        """
        from raschpy.slm import SLM
        from raschpy.pcm import PCM
        from raschpy.rsm import RSM

        gen_model = self.generating.model
        if gen_model == "SLM":
            probe = SLM(self.responses)
            probe.items = self.generating.items
        elif gen_model == "PCM":
            probe = PCM(self.responses, max_score_vector=self.generating.max_score_vector)
            probe.thresholds_uncentred = self.generating.thresholds_uncentred
        elif gen_model == "RSM":
            probe = RSM(self.responses, max_score=self.generating.max_score)
            probe.items = self.generating.items
            probe.thresholds = self.generating.thresholds
        else:
            raise ValueError(
                f"Unrecognised self.generating.model={gen_model!r} for "
                "person pseudo-estimate construction."
            )
        probe.person_estimates()
        return probe.persons

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
            zorder=3,
        )
        if len(dropped) > 0:
            ax.scatter(
                table.loc[dropped, "Anchor"],
                table.loc[dropped, "Observed"],
                s=40,
                alpha=0.7,
                edgecolors="k",
                label="Dropped",
                zorder=3,
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

    # ------------------------------------------------------------------
    # Global fit: parameter invariance across person groups
    # ------------------------------------------------------------------

    _GLOBAL_FIT_METHODS = ("wald", "andersen-lr")
    _GLOBAL_FIT_SPLITS = ("score", "person_location", "random", "exogenous")

    def _global_fit_groups(self, split, covariate, seed, split_prop=0.5):
        """Two person groups (index objects) for global_fit()."""
        resp = self.responses
        if split in ("score", "person_location"):
            if split == "score":
                value = resp.mean(axis=1)   # mean observed score: comparable under missing data
            else:
                if getattr(self, "persons", None) is None:
                    self.person_estimates()
                value = self.persons.reindex(resp.index)
            # Outer groups of split_prop each, by rank; ties (common for raw
            # scores) are broken by row order, so groups have equal sizes.
            order = value.rank(method="first", na_option="keep")
            n = order.notna().sum()
            low = order <= split_prop * n
            high = order > (1 - split_prop) * n
            return {"low": resp.index[low.values], "high": resp.index[high.values]}
        if split == "random":
            draw = np.random.default_rng(seed).random(len(resp)) < 0.5
            return {"A": resp.index[draw], "B": resp.index[~draw]}
        if getattr(self, "exogenous", None) is None or covariate is None:
            raise ValueError("split='exogenous' requires exogenous data and covariate=.")
        values = self.exogenous[covariate].reindex(resp.index)
        levels = sorted(values.dropna().unique(), key=str)
        if len(levels) != 2:
            raise ValueError(
                f"split='exogenous' needs a covariate with exactly two values; "
                f"{covariate!r} has {len(levels)}."
            )
        return {str(lv): resp.index[(values == lv).values] for lv in levels}

    def global_fit(self, method="wald", split="score", covariate=None, welch=True,
                   alpha=0.05, seed=None, split_prop=0.5,
                   se_correction="kauermann-carroll"):
        """
        Global test of parameter invariance across two groups of persons.

        Under the Rasch model, item (and threshold) parameters are the same in
        any subgroup of persons. The data are split into two groups, each
        group is calibrated separately, and the two sets of estimates are
        compared.

        method='wald' (default): W = d' (V_1 + V_2)^+ d, where d is the
        difference between the groups' estimates (re-centred, since each
        group's origin is arbitrary) and V_g each group's sandwich covariance.
        Only parameters estimable in both groups are compared: thresholds
        next to a score category unused in either group, and (RSM) category
        widths that need a fallback estimator in either group, are excluded
        and reported. With welch=True (default), W/df is referred to an F
        distribution with denominator degrees of freedom from the smaller
        group (the joint-test analogue of the Welch correction), otherwise W
        to chi-square. The sandwich covariances use the Kauermann-Carroll
        small-sample correction by default (se_correction): with few persons
        per parameter, the uncorrected sandwich is too small and the test
        liberal (PCM, about 300 persons per group: mean W/df 1.15, rejection
        rate 7.8% at alpha = 0.05, 500 null datasets); corrected, 1.00 and
        4.0%. On the same designs, the n/(n - p) correction overcorrected
        random splits (1.8%) and Mancl-DeRouen the score split (2.8%). Across
        SLM, RSM and PCM (N = 600 and 2,000; score, random and exogenous
        splits) the corrected test's mean W/df was 0.98-1.05 and its
        rejection rate 4-7%, except PCM with about 300 persons per group on
        an exogenous split (10%: the spread of W/df, not its mean, exceeds
        the F reference's with many parameters per person). Random
        splits have essentially no power against misfit (both halves contain
        the same mix of persons), so they serve as a calibration check; power
        comes from splits correlated with the misfit -- the score split for
        discrimination differences, an exogenous variable for DIF.

        method='andersen-lr': Andersen's likelihood ratio test. Reserved for
        conditional maximum likelihood (CML) estimates, not yet available, and
        raises NotImplementedError: with PAIR/CPAT estimates, which do not
        maximise the likelihood the statistic evaluates, it was not calibrated
        in simulation: it floored at 0 on score and person-location splits,
        and on exogenous splits was conservative for RSM and PCM (rejection
        rate 0-3% at 0.05), with a null distribution that no rescaling makes
        chi-square (RSM: exactly 0 in 42-46% of datasets; PCM: variance 25%
        above a scaled chi-square's).

        Parameters
        ----------
        method : {'wald', 'andersen-lr'}, default 'wald'
            'andersen-lr' raises NotImplementedError until CML is available.
        split : {'score', 'person_location', 'random', 'exogenous'}, default 'score'
            How persons are split: at the median mean observed score
            (Andersen's split; the mean handles missing data), at the median
            person location, at random, or by a two-valued exogenous
            covariate.
        covariate : str or None, default None
            Column of self.exogenous, for split='exogenous'.
        split_prop : float, default 0.5
            For split='score' or 'person_location': the proportion of persons
            in each of the two outer groups compared. 0.5 is the median split
            (everyone); 1/3 compares the lowest and highest thirds, leaving out
            the middle third; 0.4 the lowest and highest 40%. Smaller values
            give groups further apart in ability (a stronger contrast for
            discrimination differences) but fewer persons and more extreme
            targeting in each group.
        welch : bool, default True
            F reference with denominator degrees of freedom from the smaller
            group (Wald only).
        alpha : float, default 0.05
            Significance level for the decision.
        seed : int or None, default None
            Seed for split='random'.
        se_correction : {'kauermann-carroll', None}, default 'kauermann-carroll'
            Small-sample correction of each group's sandwich covariance (Wald
            only): each person's score contribution u_n is replaced by
            (I - H_n)^(-1/2) u_n, H_n being the person's leverage (Kauermann &
            Carroll, 2001; Bell & McCaffrey, 2002). Negligible with many
            persons per parameter. None uses the uncorrected sandwich.

        Attributes set
        --------------
        global_fit_summary : pandas.Series
            Method, split, statistic, df, (denominator df), p-value, decision.
        global_fit_table : pandas.DataFrame or None
            Wald only: each compared parameter's difference, SE and z.
        global_fit_excluded : list
            Wald only: parameters not compared (not estimable in both groups).
        global_fit_groups : dict
            {group name: calibrated model}.
        """
        if method not in self._GLOBAL_FIT_METHODS:
            raise ValueError(f"method must be one of {self._GLOBAL_FIT_METHODS}, got {method!r}.")
        if split not in self._GLOBAL_FIT_SPLITS:
            raise ValueError(f"split must be one of {self._GLOBAL_FIT_SPLITS}, got {split!r}.")
        if not hasattr(self, "items"):
            self.calibrate()

        if method == "andersen-lr":
            # Andersen's LR needs estimates that maximise the likelihood it
            # evaluates (CML). PAIR/CPAT estimates do not: under no misfit the
            # statistic floored at 0 on score/person-location splits, and on
            # exogenous splits was conservative for RSM/PCM, with a null
            # distribution that is not a scaled chi-square (RSM: 0 in 42-46%
            # of datasets), so no rescaling calibrates it. Kept for the
            # planned CML estimator.
            raise NotImplementedError(
                "method='andersen-lr' needs conditional maximum likelihood (CML) "
                "estimates, which are not yet available; with PAIR/CPAT estimates "
                "the likelihood ratio is not calibrated (in simulation it often "
                "floored at 0, and its null distribution was not a rescaled "
                "chi-square). Use method='wald' (the default), which is "
                "calibrated for every split."
            )

        if se_correction not in ("kauermann-carroll", None):
            raise ValueError(
                f"se_correction must be 'kauermann-carroll' or None, got {se_correction!r}."
            )
        if not 0 < split_prop <= 0.5:
            raise ValueError(f"split_prop must be in (0, 0.5], got {split_prop!r}.")
        groups = self._global_fit_groups(split, covariate, seed, split_prop)
        names = list(groups)
        fitted, parts = {}, []
        for name in names:
            m = self._global_fit_subset(groups[name])
            fitted[name] = m
            parts.append(self._global_fit_params(m, se_correction is not None))
        (lab, e1, V1, ok1, blocks), (_, e2, V2, ok2, _) = parts
        keep = ok1 & ok2
        idx = np.flatnonzero(keep)
        C = np.eye(len(idx))
        for blk in blocks:   # re-centre each location block over the kept entries
            pos = np.flatnonzero(np.isin(idx, blk))
            if len(pos):
                C[np.ix_(pos, pos)] -= 1.0 / len(pos)
        d = C @ (e1 - e2)[idx]
        S = C @ (V1 + V2)[np.ix_(idx, idx)] @ C.T
        W = float(d @ np.linalg.pinv(S, rcond=1e-10) @ d)
        df = int(np.linalg.matrix_rank(S, tol=1e-10 * np.abs(S).max()))
        n_small = min(len(groups[n]) for n in names)
        if welch:
            ddf = max(n_small - df, 1)
            p = float(f_dist.sf(W / df, df, ddf))
        else:
            ddf = None
            p = float(chi2_dist.sf(W, df))
        se = np.sqrt(np.clip(np.diag(S), 0, None))
        kept_labels = [lab[i] for i in idx]
        self.global_fit_table = pd.DataFrame(
            {f"{names[0]}": (C @ e1[idx]), f"{names[1]}": (C @ e2[idx]),
             "Difference": d, "SE": se, "z": np.divide(d, se, out=np.full_like(d, np.nan), where=se > 0)},
            index=pd.Index(kept_labels, name="Parameter"))
        self.global_fit_excluded = [lab[i] for i in np.flatnonzero(~keep)]
        self.global_fit_groups = fitted
        summary = {"method": method, "split": split, "statistic": W, "df": df}
        if split in ("score", "person_location"):
            summary["split_prop"] = split_prop
        if welch:
            summary["denominator df"] = ddf
        summary["se_correction"] = se_correction
        summary.update({"p-value": p, "reject": bool(p < alpha),
                        "group sizes": {n: len(groups[n]) for n in names},
                        "parameters excluded": len(self.global_fit_excluded)})
        self.global_fit_summary = pd.Series(summary, name="Global fit")
        if self.global_fit_excluded:
            warnings.warn(
                f"{len(self.global_fit_excluded)} parameter(s) not compared because "
                f"they are not estimable in both groups (e.g. next to a score "
                f"category unused in one group): {self.global_fit_excluded[:10]}"
                f"{' ...' if len(self.global_fit_excluded) > 10 else ''}. See "
                f"global_fit_excluded.",
                UserWarning,
                stacklevel=2,
            )
        return self.global_fit_summary

    def wright_map(
        self,
        model=None,
        person_names=None,
        item_names=None,
        facet_element_names=None,
        expand_model=False,
        show_stretch=True,
        item_level="items",
        threshold_type="rasch-andrich",
        orientation="vertical",
        map_type="hist",
        item_labels=False,
        item_strip=False,
        item_distribution=False,
        facet_labels=False,
        overlay_facet=False,
        strip_thickness=2.0,
        strip_alpha=0.3,
        sort="location",
        palette="colorblind multi",
        neutral_extremes=False,
        item_row_height=None,
        facet_row_height=None,
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
        facet_lim=None,
        person_scaling=None,
        facet_scaling=1,
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
        facet_color="mediumseagreen",
        person_line_color="black",
        item_line_color="black",
        facet_line_color="black",
        marker_color=None,
        alpha=0.6,
        black=False,
        font="Times New Roman",
        title_font_size=15,
        axis_font_size=12,
        labelsize=12,
        item_label_size=8,
        facet_label_size=8,
        tick_interval=1,
        filename=None,
        file_format="png",
        dpi=300,
    ):
        """
        Plot a Wright map showing person and item location distributions
        (and, on a facet-element model such as MFRM, a third facet-element
        distribution alongside them).

        Displays the distribution of person locations alongside item
        difficulties on the same logit scale. Items can either mirror the
        person distribution on the opposite side of a zero baseline (as
        back-to-back histograms or KDE curves), or be listed by name in a
        dedicated panel, Winsteps-style, aligned to the same location axis
        (item_labels=True) -- or, on a model with threshold/category
        structure (any model other than a dichotomous one such as SLM),
        as a divided category-width strip (item_strip=True).

        On a facet-element model (checked via a self.facets attribute, not
        an isinstance check, so any future facet-element model class picks
        this up for free -- MFRM is currently the only one), a `model`
        argument selects which facet element (e.g. rater) sub-model to plot,
        and a facet-element panel is added alongside persons and items,
        either as its own dedicated panel (default) or overlaid onto the
        items panel/axis (overlay_facet=True). The facet named 'raters' by
        default is whatever self.facets/self.facet_names alias to for this
        instance (e.g. 'judges', if constructed with a custom facet name).
        `model` and every facet-specific argument below (facet_labels,
        overlay_facet, expand_model, show_stretch, facet_element_names,
        and the facet_*/*_facet display arguments) are unavailable on a
        model with no facet structure (SLM/PCM/RSM) -- `model` must stay
        None there, and facet_labels/overlay_facet raise ValueError if set.

        Parameters
        ----------
        model : str or None, default None
            Facet-element (facet element) sub-model: 'global', 'items',
            'thresholds', 'matrix', 'bivector', 'centrality', 'pseudo_halo',
            or 'bistretch'. Only meaningful on a facet-element model such as
            MFRM (checked via a self.facets attribute, not an isinstance
            check, so any future facet-element model class picks this up
            for free) -- required there (raises ValueError if left None, so
            the model actually plotted always matches what was passed,
            rather than silently falling back to 'global' and, if that
            hasn't been calibrated yet, calibrating it fresh); must stay
            None on SLM/PCM/RSM, which have no facet structure at all
            (raises ValueError if given a value there). Prefer the
            model-specific aliases (wright_map_global, wright_map_items,
            wright_map_thresholds, wright_map_matrix, wright_map_bivector,
            wright_map_centrality, wright_map_pseudo_halo,
            wright_map_bistretch) when convenient. For 'items'/
            'thresholds', each facet element's own multiple locations
            (one per item, or one per threshold) are all plotted
            individually, named e.g. 'Rater_1_Item_1'/'Rater_1_1'. For
            'matrix', every (facet element, item, threshold) combination
            is plotted individually, named e.g. 'Rater_1_Item_1_1' --
            this can be a *lot* of individual values (facet elements x
            items x max_score), so facet_labels=True in particular can
            produce a very tall panel there. For 'bivector', its own two
            marginal location vectors (per item, per threshold) are
            plotted directly by default rather than the full (facet
            element, item, threshold) grid they're summed to reconstruct
            -- see expand_model below to plot that full grid instead.
            Named the same way as 'items'/'thresholds' (e.g.
            'Rater_1_Item_1' and 'Rater_1_1' both appearing for the same
            facet element). For the stretch models (centrality, pseudo_halo,
            bistretch), each is a 2-(or 3-)parameter (lambda shift,
            omega stretch[es]) restriction of an unrestricted model --
            the facet panel plots lambda (a plain logit-scale shift, one
            value per facet element) exactly as 'global' would by default; see
            expand_model below to plot each facet element's full reconstructed
            vector instead. omega isn't a logit value at all either way
            (a dimensionless spread multiplier centred on 1), so it
            never appears on this axis -- see show_stretch below for the
            separate panel(s) it gets instead.
        person_names : str, list, or None, default None
            Person subset to include. None uses all persons.
        item_names : str, list, or None, default None
            Item subset to include. None uses all items.
        facet_element_names : str, list, or None, default None
            Facet-element subset to include (e.g. specific facet elements). None
            uses all elements of the facet.
        expand_model : bool, default False
            Only used for model='bivector' or one of the stretch models
            (centrality, pseudo_halo, bistretch; silently ignored for
            every other model). If True, plots each facet element's full
            reconstructed vector instead of its single marginal/lambda
            point -- shaped differently depending on model:
              - 'bivector': the full (facet element, item, threshold)
                grid, from summing its own two marginal location vectors
                -- every entry is just one item-vector value plus one
                threshold-vector value, not an independent estimate, but
                this can still be useful to see how the two vectors
                interact at each individual (item, threshold)
                combination. Named the same way as 'matrix', e.g.
                'Rater_1_Item_1_1'.
              - 'centrality': the per-threshold vector lambda + (omega -
                1) * tau_k reconstructs, one point per Rasch-Andrich
                threshold, named the same way 'thresholds' would be
                (e.g. 'Rater_1_1').
              - 'pseudo_halo': the per-item vector lambda + (omega - 1) *
                delta_i reconstructs, one point per item, named the same
                way 'items' would be (e.g. 'Rater_1_Item_1').
              - 'bistretch': the per-(item, threshold) vector both
                reconstructed axes sum to, named the same way 'matrix'
                would be (e.g. 'Rater_1_Item_1_1').
            The stretch panel(s) below (show_stretch) are unaffected
            either way -- they show omega itself, not a location, so
            there's nothing to expand there.
        show_stretch : bool, default True
            Only used for the stretch models (centrality, pseudo_halo,
            bistretch; silently ignored for every other model). Adds
            omega -- each facet element's stretch factor(s) -- as its own
            panel(s) bolted onto the right of the whole figure, one bar
            per facet element in plain facet-name order, centred on 1 (>1
            expands, <1 compresses; see each model's own calibrate_*
            docstring for what that means there) rather than 0, on its
            own scale entirely unrelated to the shared Location axis.
            bistretch gets two such panels stacked (omega_items above
            omega_thresholds), sharing the same facet element order and the same
            y-limits as each other (so a bar's length is comparable
            panel to panel, not just within its own). Set False for a
            plain shift-only (lambda) map, styled exactly like
            model='global'.
        item_level : str, default 'items'
            'items' plots one point per item at its central location.
            'thresholds' plots one point per Rasch-Andrich threshold
            instead (RSM/MFRM: delta_i + tau_k; PCM: thresholds_uncentred),
            flattened across all items. Ignored when item_strip=True,
            which always needs each item's full threshold set regardless
            of this setting. 'thresholds' and item_strip=True both require
            threshold/category structure -- raises ValueError on a
            dichotomous model (SLM), which has none.
        threshold_type : str, default 'rasch-andrich'
            Which threshold convention to plot when item_level='thresholds'.
            'rasch-andrich' (default) plots the raw model parameters
            (thresholds_uncentred for models with per-item threshold
            structure, e.g. PCM; item_location + tau_k for models sharing
            one threshold vector, e.g. RSM/MFRM -- ignoring facet/rater
            effects for MFRM, same as the existing point positions), the
            point where categories k-1 and k are equally likely -- these
            can be disordered. 'thurstonian' plots the median (cumulative)
            threshold instead: the location where P(X >= k) = 0.5, found
            by root-finding -- always ordered, regardless of the
            underlying Rasch-Andrich thresholds. 'half-point' plots the
            location where the expected item score E[X] = k - 0.5, found
            by root-finding -- also always ordered. Both alternatives use
            the same baseline as 'rasch-andrich' (facet/rater-free for
            MFRM, consistent across all 8 model sub-types there). Ignored
            when item_level='items'. Also governs item_strip=True's own
            category-boundary segments: 'rasch-andrich' draws the
            most-probable-category regions (the upper envelope of each
            category's response-probability curve, which can drop a
            category entirely if its Rasch-Andrich thresholds are
            disordered -- a UserWarning is raised in that case).
            'thurstonian'/'half-point' instead draw the plain intervals
            between consecutive (guaranteed monotonic) boundaries --
            disorder cannot occur, so no warning is possible for these two.
        orientation : str, default 'vertical'
            'vertical' places location on the x-axis, persons above the
            baseline. 'horizontal' places location on the y-axis, persons
            to the left of the baseline.
        map_type : str, default 'hist'
            'hist' plots (back-to-back, when item_labels=False) histograms.
            'kde' plots smoothed kernel density estimate curves instead.
            Only governs the person side once item_labels=True, since the
            item side becomes a label panel rather than a distribution.
            Also governs the facet panel, when it is a distribution rather
            than facet_labels.
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
            If True, adds a further panel -- a plain (non-mirrored) hist or
            KDE of item (or threshold, per item_level) locations -- between
            the person panel and the item_labels panel. Implies
            item_labels=True.
        facet_labels : bool, default False
            Requires model (raises ValueError otherwise -- there is no
            facet distribution to label without one). If True, facet
            elements are listed by name in a dedicated panel (grouped into
            no_of_bins location bins, stacked as rows within a bin, ordered
            by location), the same way item_labels lists items, rather
            than shown as a hist/kde distribution. Forces overlay_facet=False.
        overlay_facet : bool, default False
            Requires model (raises ValueError otherwise). If True, the
            facet distribution is overlaid (as an outlined, hatched shape
            rather than a filled one) onto the items panel or axis instead
            of getting its own dedicated panel. Silently forced to False
            whenever item_labels=True and/or facet_labels=True, since
            overlaying a distribution onto a name-label panel doesn't make
            visual sense.
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
        facet_row_height : float or None, default None
            Physical height, in inches, allocated per stacked facet-label
            row. If None, measured automatically from the longest facet
            element name at the given font size. Only used when
            facet_labels=True.
        narrow_font : bool, default False
            If True, item labels, facet-element labels, and (with
            item_strip=True) the row labels beside the strip are set in
            'Arial Narrow' instead of font. A condensed font renders each
            name shorter, so it needs less stacked row height -- useful
            when names are long and many end up sharing a bin or crowding
            the strip. Falls back to matplotlib's usual font substitution
            if 'Arial Narrow' isn't installed. Only affects item_labels=True
            and/or facet_labels=True; everything else (axis text, legend,
            title) still uses font.
        edge_padding : float, default 0.5
            Extra padding, in inches, added around the item-label margin
            (left of the axes in vertical orientation, below the axes in
            horizontal orientation) beyond what's needed to fit the
            longest item name. The same value produces matching-looking
            margins in both orientations. Only used when item_strip=True
            (facet_labels' own binned panel, like item_labels' own,
            doesn't need this margin -- names are drawn inside the panel,
            not as outside row labels).
        distribution_markers : bool, default False
            If True, overlays mean/+-1SD/+-2SD reference marks (labelled
            mu, mu+-sigma, mu+-2sigma) for persons and items, in their
            respective colours.
        group_by : pandas.Series or None, default None
            Person-level grouping (e.g. a DIF group), indexed like
            self.persons_{model}, used to split the person distribution
            into one sub-distribution per unique value. None plots persons
            as a single series.
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
            One-sided magnitude for persons' own Count/Proportion axis
            (e.g. person_lim=30 shows 0 to 30). If None, chosen
            automatically from persons' own natural peak, independently
            of items'/facets' own scale. Persons' own panel always keeps
            its full physical size regardless of this value -- items'
            and facets' own panels (when they get a dedicated hist/kde
            panel, i.e. item_distribution=True and/or a non-facet_labels,
            non-overlaid facet panel) are sized proportionally against
            it, so a given Count/Proportion value reads at the same
            physical scale in every panel that participates.
        item_lim : float or None, default None
            One-sided magnitude for items' own Count/Proportion axis.
            Same semantics as person_lim, for whichever panel items end
            up in (mirrored onto the same axis as persons when
            item_labels=False, or their own item_distribution panel).
        facet_lim : float or None, default None
            One-sided magnitude for the facet-element distribution's own
            Count/Proportion axis. Same semantics as person_lim, for
            whichever panel the facet distribution ends up in (overlaid
            onto the items panel/axis when overlay_facet=True, or its
            own dedicated panel). Only used when the facet is shown as a
            distribution rather than facet_labels.
        person_scaling : float or None, default None
            Rebalances the item (and facet) distribution against the
            person distribution. None chooses it automatically: the
            largest of 1, 2, 5, 10, 20, 50, ... that keeps the rescaled
            item (and facet) distribution within half the height of the
            person distribution, and never below 1. With
            person_scaling=10 they read 10x
            larger per Count/Proportion unit than the person side, so
            they take 10x more of the plot. Useful when far more persons
            than items/facet elements otherwise leave those distributions
            a flat, uninformative squiggle at the persons' scale. All
            sides keep their true tick-label values. When items (and, not
            overlaid, the facet) have their own panel -- item_labels=True
            or item_distribution=True -- that panel is grown to match,
            enlarging the figure. In the mirrored single-axis map
            (item_labels=False) the zero baseline instead shifts to give
            the item side (and an overlaid facet) more of the shared
            axis, at the same figure size -- the person side is
            compressed to make room. See person_lim for why the
            comparison isn't well-defined when items are shown only as
            labels with no distribution.
        facet_scaling : float, default 1
            Reduces the facet panel's own physical size by this factor,
            on top of person_scaling -- independent of it, so the facet
            panel can be dialled back without changing the item panel.
            facet_scaling=10 makes the facet panel 10x smaller than it
            would otherwise be. Useful for the extended MFRM models
            (items, thresholds, matrix, bivector), where the facet
            panel's natural range comes from far more individual values
            than persons'/items' (e.g. matrix's facet_elements x items x
            max_score raw locations), which can otherwise make it
            balloon out of proportion even at person_scaling=1 -- and
            gets worse still if person_scaling != 1, since that
            multiplier applies to the facet panel too. facet_scaling=1
            leaves that behaviour as-is.
        no_of_bins : int, default 20
            Number of histogram bins spanning the location range. Also
            defines the location-binning grid used for item_labels=True
            and facet_labels=True.
        pad : bool, default False
            If True, adds 5% padding to either end of the location axis
            limits. If False, the location axis is bounded exactly to the
            plotted range.
        plot_range : tuple of (float, float) or None, default None
            (lo, hi) limits for the location axis. If None, uses the floor
            of the combined minimum and the ceiling of the combined maximum
            of persons, items, and facet elements. When item_strip=True,
            this auto range is additionally widened, if needed, so that
            every item's outermost category (its lowest and highest
            category segments, which are otherwise unbounded) shows at
            least 1 logit of visible width rather than being clipped to
            nothing at the axis edge.
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
            to fit however many stacked item and/or facet-element rows are
            needed -- the person distribution's own size is never reduced
            to make room.
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
        facet_color : str, default 'mediumseagreen'
            Fill colour for the facet-element distribution. Only used when
            facet_labels=False. Overridden by a grey shade when black=True.
        person_line_color : str, default 'black'
            Line colour for the person KDE curve. Only used when
            map_type='kde' and group_by is None.
        item_line_color : str, default 'black'
            Colour for item labels (item_labels=True) or the item KDE
            curve line (item_labels=False, map_type='kde').
        facet_line_color : str, default 'black'
            Colour for facet-element labels (facet_labels=True) or the
            facet KDE curve line (facet_labels=False, map_type='kde').
        marker_color : str, tuple or None, default None
            Colour of the tick marks and labels drawn when
            distribution_markers=True. None gives each set of marks a
            darker shade of its own distribution's colour (person marks
            from person_color, item/threshold marks from item_color). A
            single colour applies to both; a (person, item) pair sets them
            separately. Person marks use this only for the single-series
            (group_by=None) case -- grouped marks use each group's own
            colour instead, to stay identifiable against its distribution.
        alpha : float, default 0.6
            Fill transparency for the distributions.
        black : bool, default False
            If True, renders person_color/item_color/facet_color as grey
            shades instead. Has no effect on group_by colours.
        font : str, default 'Times New Roman'
            Font family for all plot text.
        title_font_size : int, default 15
            Title font size in points.
        axis_font_size : int, default 12
            Axis label font size in points.
        labelsize : int, default 12
            Tick label font size in points.
        item_label_size : int, default 8
            Font size, in points, for item names in the item_labels panel.
        facet_label_size : int, default 8
            Font size, in points, for facet-element names in the
            facet_labels panel.
        tick_interval : float, default 1
            Spacing between Location axis ticks. Ticks are evenly spaced
            at multiples of this value across the plotted range, rather
            than using matplotlib's own automatic tick choice.
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
        # model dispatch (facet-element/rater structure) is an MFRM-only
        # concept -- SLM/PCM/RSM have no facets attribute at all, and pass
        # model=None (the shared default). An MFRM instance always needs a
        # real model, since it has no single unparameterised rater
        # representation to fall back on.
        has_facets = hasattr(self, "facets")
        if has_facets and model is None:
            raise ValueError(
                "model is required for wright_map on a many-facet "
                "model such as MFRM -- e.g. model='global'."
            )
        if not has_facets and model is not None:
            raise ValueError(
                f"model is only meaningful for a many-facet model "
                f"such as MFRM -- {type(self).__name__} has no facet structure "
                f"to select {model!r} from."
            )
        if has_facets and model not in (
            "global", "items", "thresholds", "matrix", "bivector",
            "centrality", "pseudo_halo", "bistretch",
        ):
            raise NotImplementedError(f"wright_map does not support the {model!r} MFRM model.")
        if not has_facets and (facet_labels or overlay_facet):
            raise ValueError(
                f"facet_labels/overlay_facet require a many-facet "
                f"model such as MFRM -- {type(self).__name__} has no facet "
                f"structure to display."
            )
        # dichotomous models (SLM) have no threshold structure at all --
        # item_level='items'/item_strip=False (both defaults) never touch
        # self.thresholds, so this only bites when someone actually asks
        # for per-threshold display on a model that can't provide it.
        if (item_level == "thresholds" or item_strip) and not hasattr(self, "thresholds"):
            raise ValueError(
                f"item_level='thresholds' and item_strip=True require "
                f"threshold structure, which {type(self).__name__} does not "
                f"have (dichotomous models have no category thresholds)."
            )

        if black:
            person_color = "lightgray"
            item_color = "darkgray"
            facet_color = "gray"

        # Distribution-marker colours: by default each set of marks takes a
        # darker shade of its own distribution's colour, so it reads as
        # belonging to that distribution.
        if marker_color is None:
            person_marker_color = _darken(person_color)
            item_marker_color = _darken(item_color)
        elif isinstance(marker_color, (tuple, list)) and len(marker_color) == 2:
            person_marker_color, item_marker_color = marker_color
        else:
            person_marker_color = item_marker_color = marker_color

        if has_facets:
            if not hasattr(self, f"persons_{model}"):
                self.person_estimates(model=model)
            all_persons = getattr(self, f"persons_{model}")
        else:
            if not hasattr(self, "persons"):
                self.person_estimates()
            all_persons = self.persons
        persons = all_persons if person_names is None else all_persons.loc[person_names]

        # a strip needs every item's full threshold set regardless of
        # item_level (it draws the whole operating range, not one point),
        # so it also implies item_labels -- friendlier than raising when
        # someone passes item_strip=True on its own
        if item_strip:
            item_labels = True

        # a plain item/threshold hist or KDE panel only makes sense as a
        # further panel alongside the labelled one -- on its own it's just
        # the old mirrored mode with item_labels=False
        if item_distribution:
            item_labels = True

        # overlaying a distribution onto a name-label panel doesn't make
        # visual sense -- always use a separate facet panel in that case
        if item_labels or facet_labels:
            overlay_facet = False

        # items get a genuine, separate, scale-matched panel only when
        # explicitly requested (item_distribution=True, which also forces
        # item_labels=True above).
        item_dist_panel = item_distribution

        # In the mirrored single-axis map (item_labels=False) persons and
        # items -- and an overlaid facet -- share one count axis, so
        # person_scaling has no separate panel to resize. Instead, exactly
        # as for SLM/RSM/PCM, it enlarges the item side of that shared
        # axis by this factor: the item distribution (and any overlaid
        # facet) ends up person_scaling x larger relative to persons, with
        # the figure size unchanged (the person side is compressed to make
        # room). 1 whenever items have their own panel, so the panel-
        # sizing use of person_scaling below is untouched. Set once
        # person_scaling itself is resolved, after the count ranges below.

        if has_facets and not hasattr(self, f"facet_effects_{model}"):
            self.calibrate(model=model)
        # a many-facet model's item locations / thresholds depend on the sub-model
        own_items = self._items_for(model) if has_facets else self.items
        own_thresholds = self._thresholds_for(model) if has_facets else getattr(self, "thresholds", None)
        base_items = own_items if item_names is None else own_items.loc[item_names]

        if has_facets:

            # facet_element_names always filters by rater name (matching
            # what it does for 'global'), applied before flattening below --
            # for 'items'/'thresholds' that's the raw table's own row index;
            # for 'matrix' it's the first level of its (rater, item) row
            # index, which .loc[] with a plain list of rater names also
            # selects on
            def _filter_by_rater(table):
                return table if facet_element_names is None else table.loc[facet_element_names]

            def _flatten_facets(table):
                # one row per rater, one column per item/threshold/category
                # -- stacking flattens that to one entry per combination,
                # e.g. 'Rater_1_Item_1' or 'Rater_1_1' for a 2-level table,
                # 'Rater_1_Item_1_1' for a 3-level one (matrix's own (rater,
                # item) row index plus the stacked category column)
                stacked = table.stack().dropna()
                return stacked.set_axis(
                    ["_".join(str(p) for p in idx) for idx in stacked.index]
                )

            # The stretch models (centrality, pseudo_halo, bistretch) are each
            # a 2-(or 3-)parameter (lambda shift, omega stretch[es])
            # restriction of an unrestricted model -- lambda is a plain
            # logit-scale shift, one value per rater, structurally identical
            # to facet_effects_global, so it plots exactly like 'global' by
            # default. omega (a dimensionless spread multiplier centred on 1,
            # not a logit value) has no meaningful place on the shared
            # Location axis at all either way -- see the bolted-on stretch
            # panel(s) near the end of this method instead (show_stretch=
            # True). expand_model=True swaps facets for the full
            # reconstructed per-threshold (centrality)/per-item (pseudo_halo)/
            # per-(item, threshold) (bistretch) vector instead of the single
            # lambda point -- lambda + omega combined back out to individual
            # element positions, named the same way the corresponding
            # unrestricted model (thresholds/items/matrix) would be. The
            # omega panel(s) below are unaffected either way, since they show
            # what drove the expansion, not a location. Same argument
            # bivector's own expansion below uses -- each model resolves it
            # to whatever shape (flat per-element vector here; a (facet
            # element, item, threshold) grid there) its own restriction
            # implies.
            stretch_omega = None
            if model in ("centrality", "pseudo_halo", "bistretch"):
                fe_stretch = getattr(self, f"facet_effects_{model}")
                if expand_model:
                    facets = _flatten_facets(
                        _filter_by_rater(
                            getattr(self, f"facet_effects_{model}_full_vector")
                        )
                    )
                else:
                    facets = _filter_by_rater(fe_stretch["lambda"])
                if model == "bistretch":
                    stretch_omega = {
                        "omega_items": _filter_by_rater(fe_stretch["omega_items"]),
                        "omega_thresholds": _filter_by_rater(fe_stretch["omega_thresholds"]),
                    }
                else:
                    stretch_omega = {"omega": _filter_by_rater(fe_stretch["omega"])}
            elif model == "global":
                facets = _filter_by_rater(self.facet_effects_global)
            elif model == "bivector" and not expand_model:
                # bivector's true parameters are the two marginal vectors
                # (per-item severity, per-threshold severity) its full
                # (rater, item, threshold) matrix is reconstructed FROM (as
                # their sum) -- showing that full reconstructed matrix would
                # be redundant, since every entry is just one item-vector
                # value plus one threshold-vector value, not an independent
                # estimate. Plotting the two vectors directly instead, named
                # the same way the standalone items/thresholds models are,
                # shows what bivector actually estimates. expand_model=True
                # opts into the full grid instead (silently ignored for
                # every other model, where it has no separate meaning).
                items_facets = _flatten_facets(
                    _filter_by_rater(self.facet_effects_bivector_items)
                )
                thr_facets = _flatten_facets(
                    _filter_by_rater(self.facet_effects_bivector_thresholds)
                )
                facets = pd.concat([items_facets, thr_facets])
            else:
                facets = _flatten_facets(
                    _filter_by_rater(getattr(self, f"facet_effects_{model}"))
                )
            facet_label = self.facets.capitalize()
        else:
            facets = pd.Series([], dtype=float)
            facet_label = None
            stretch_omega = None

        # the mirrored / item_distribution hist/KDE draws Rasch-Andrich
        # thresholds rather than item central locations whenever item_level
        # asks for them (or item_strip forces the per-threshold
        # flattening) -- label its legend entry to match.
        item_dist_label = (
            "Thresholds" if (item_level == "thresholds" or item_strip) else "Items"
        )

        valid_threshold_types = ("rasch-andrich", "thurstonian", "half-point")
        if threshold_type not in valid_threshold_types:
            raise ValueError(
                f"threshold_type must be one of {valid_threshold_types}, "
                f"got {threshold_type!r}."
            )

        if item_level == "items" and not item_strip:
            items = base_items
        else:
            # one entry per Rasch-Andrich threshold rather than one per item.
            # RSM/MFRM share a single threshold set (self.thresholds is a
            # Series of step values tau_k): delta_i + tau_k as one outer-sum
            # matrix. PCM has its own threshold set per item, already on the
            # shared absolute scale -- but as thresholds_uncentred, not
            # self.thresholds (which is centred *within* each item, so its
            # row means are ~0 and aren't comparable to self.items/persons
            # at all). Either way the result is a wide (item x threshold
            # number) frame.
            if isinstance(own_thresholds, pd.Series):
                matrix = pd.DataFrame(
                    base_items.values[:, None] + own_thresholds.values[None, :],
                    index=base_items.index,
                    columns=own_thresholds.index,
                )
            else:
                matrix = self.thresholds_uncentred.loc[base_items.index]

            if threshold_type != "rasch-andrich":
                # Thurstonian (median/cumulative) and half-point (expected-
                # score) thresholds have no closed form -- root-find each
                # one, item by item, on the facet/rater-free baseline above
                # (delta_i + tau_k for RSM/MFRM, thresholds_uncentred for
                # PCM) -- for MFRM this matches how this method already
                # ignores rater effects for point positions, consistent
                # across all 8 model sub-types. cat_prob_fn/exp_score_fn
                # dispatch to whichever underlying probability model this
                # instance actually has: MFRM's own rater-free softmax
                # (self.cat_prob/self.exp_score take a facet_element and
                # facet_effects argument neither has a meaning for here, so
                # this method keeps its own local, facet-free reimplementation
                # rather than trying to force them through that signature);
                # RSM's self.cat_prob/self.exp_score (item_location explicit);
                # or PCM's self.cat_prob_uncentred/self.exp_score_uncentred
                # (item_location already baked into the per-item threshold
                # vector, not a separate argument). Column labels (k) are
                # kept from the Rasch-Andrich matrix so the two conventions
                # stay directly comparable at the same nominal threshold
                # number. Used for both the flattened point representation
                # and (below) the item_strip segments -- unlike
                # Rasch-Andrich thresholds, these are guaranteed monotonic,
                # so item_strip's boundaries follow them directly with no
                # most-probable-category envelope needed and no
                # disordered-thresholds warning.
                if has_facets:
                    def cat_prob_fn(theta, item_location, category, thresh):
                        cats = np.arange(len(thresh) + 1, dtype=float)
                        cumsum = np.concatenate([[0.0], np.cumsum(thresh)])
                        log_nums = cats * (theta - item_location) - cumsum
                        log_nums -= log_nums.max()
                        nums = np.exp(log_nums)
                        return nums[category] / nums.sum()

                    def exp_score_fn(theta, item_location, thresh):
                        cats = np.arange(len(thresh) + 1, dtype=float)
                        cumsum = np.concatenate([[0.0], np.cumsum(thresh)])
                        log_nums = cats * (theta - item_location) - cumsum
                        log_nums -= log_nums.max()
                        nums = np.exp(log_nums)
                        probs = nums / nums.sum()
                        return (cats * probs).sum()
                elif isinstance(own_thresholds, pd.Series):
                    def cat_prob_fn(theta, item_location, category, thresh):
                        return self.cat_prob(theta, item_location, category, thresh)

                    def exp_score_fn(theta, item_location, thresh):
                        return self.exp_score(theta, item_location, thresh)
                else:
                    def cat_prob_fn(theta, item_location, category, thresh):
                        return self.cat_prob_uncentred(theta, category, thresh)

                    def exp_score_fn(theta, item_location, thresh):
                        return self.exp_score_uncentred(theta, thresh)

                display_matrix = pd.DataFrame(
                    index=matrix.index, columns=matrix.columns, dtype=float
                )
                for item in matrix.index:
                    row = matrix.loc[item].dropna()
                    m = len(row)
                    item_location = base_items.loc[item]
                    if isinstance(own_thresholds, pd.Series):
                        thresh = own_thresholds.loc[row.index].values
                    else:
                        thresh = row.values

                    lo, hi = row.min() - 15, row.max() + 15
                    for k in row.index:
                        if threshold_type == "thurstonian":
                            def target(theta, k=k, m=m):
                                return sum(
                                    cat_prob_fn(theta, item_location, c, thresh)
                                    for c in range(k, m + 1)
                                ) - 0.5
                        else:  # half-point
                            def target(theta, k=k):
                                return exp_score_fn(theta, item_location, thresh) - (k - 0.5)
                        display_matrix.loc[item, k] = brentq(target, lo, hi)
                matrix_for_points = display_matrix
            else:
                matrix_for_points = matrix

            # pandas' stack() stopped dropping NaN by default as of the
            # 2.1+ implementation, so drop the short items' padding NaNs
            # explicitly rather than relying on stack()'s own default
            stacked = matrix_for_points.stack().dropna()
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
                # exactly the interval between adjacent thresholds; when
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
                    item: row.dropna().values
                    for item, row in (
                        matrix if threshold_type == "rasch-andrich" else display_matrix
                    ).iterrows()
                }
                if threshold_type == "rasch-andrich":
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
                else:
                    # Thurstonian/half-point boundaries are each the root of
                    # a strictly monotonic function of theta, so they are
                    # guaranteed already in increasing order -- segments are
                    # simply the intervals between consecutive boundaries,
                    # no most-probable-category envelope needed and no
                    # disordered-thresholds case can arise.
                    item_segments = {}
                    for item, boundaries in item_thresholds_natural.items():
                        m = len(boundaries)
                        item_segments[item] = [
                            (
                                None if i == 0 else boundaries[i - 1],
                                None if i == m else boundaries[i],
                                i,
                            )
                            for i in range(m + 1)
                        ]
                    disordered_items = []

        no_of_items = len(items)
        no_of_facets = len(facets)

        if group_by is not None:
            group_by = group_by.reindex(persons.index)
            person_group_values = sorted(group_by.dropna().unique(), key=str)
            if group_colors is None:
                cmap = plt.get_cmap("tab10")
                group_colors = {
                    g: cmap(i % 10) for i, g in enumerate(person_group_values)
                }

        if plot_range is None:
            combined = np.concatenate([persons.values, items.values, facets.values])
            span_lo, span_hi = float(combined.min()), float(combined.max())
            if item_strip:
                # item_strip's own segment boundaries (built above, from
                # item/threshold locations directly) can fall well outside
                # the persons/items/facets span -- e.g. a wide bottom
                # category on an easy item -- and get clipped to nothing
                # at the axis edge otherwise. Fold in the boundary between
                # the lowest two categories (and the highest two) across
                # all items, with a guaranteed 1-logit margin beyond it,
                # so every item's outermost category always shows at
                # least a 1-logit sliver rather than vanishing entirely.
                lowest_bounds = [
                    segs[0][1] for segs in item_segments.values() if len(segs) > 1
                ]
                highest_bounds = [
                    segs[-1][0] for segs in item_segments.values() if len(segs) > 1
                ]
                if lowest_bounds:
                    span_lo = min(span_lo, min(lowest_bounds) - 1.0)
                if highest_bounds:
                    span_hi = max(span_hi, max(highest_bounds) + 1.0)
            if distribution_markers:
                # the μ±2σ ticks (and their rotated labels) are drawn at
                # mean ± 2·SD of each plotted distribution and can fall
                # outside its raw data extent when that distribution is
                # wide or heavy-tailed — fold them into the range so they
                # stay on-canvas rather than being clipped at the spine
                marker_values = [items.values, facets.values]
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

        # each distribution's own natural peak, computed analytically
        # (no plotting needed) so panel sizing/axis ranges can be
        # resolved before the figure is even built. Computed
        # independently per distribution -- deliberately NOT sharing a
        # single "widest" value across persons/items/facets, since a
        # small-N facet's own much more sharply peaked KDE would
        # otherwise force persons' own well-populated distribution onto
        # a needlessly inflated scale (and vice versa for a tiny facet
        # squeezed onto persons' own wide one)
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
        facet_natural_max = (
            _natural_max(facets.values, no_of_facets) if has_facets else 0
        )

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
        if facet_lim is not None:
            facet_range = facet_lim
            facet_step = _nice_step_for(facet_lim, _integer_counts)
        else:
            facet_range, facet_step = _nice_ceil_and_step(
                facet_natural_max, _integer_counts
            )

        # person_scaling=None (the default) picks the rebalancing factor
        # automatically, so a map with far more persons than items (or
        # facet elements) doesn't leave those distributions a flat band at
        # the persons' scale: the largest 1/2/5 x 10^k value keeping each
        # rescaled distribution's extent within half the persons' own,
        # never below 1. Only distributions actually drawn at the
        # person_scaling scale count: items shown as a distribution
        # (mirrored map or item_distribution panel), and the facet when
        # it is drawn as a distribution rather than as labels.
        if person_scaling is None:
            scaled_ranges = []
            if not item_labels or item_dist_panel:
                scaled_ranges.append(item_range)
            if has_facets and not facet_labels:
                scaled_ranges.append(facet_range / facet_scaling)
            scaled_ranges = [r for r in scaled_ranges if r > 0]
            person_scaling = 1
            if scaled_ranges and person_range > 0:
                target = 0.5 * person_range / max(scaled_ranges)
                exponent = 0
                while True:
                    for base in (1, 2, 5):
                        candidate = base * 10**exponent
                        if candidate > target:
                            break
                        person_scaling = candidate
                    else:
                        exponent += 1
                        continue
                    break
        mirror_item_scale = person_scaling if not item_labels else 1

        # breathing room reserved before row 0 of the item panel (used for
        # both the boundary_margin below and, when distribution_markers
        # draws a tick alongside each item-side mark, for keeping that
        # tick inside the margin and clear of row 0's own text -- shared
        # so the two stay consistent if this value ever changes)
        item_row_margin = 0.45
        # same idea for the facet-labels panel, which never carries
        # distribution_markers' own reserved band -- kept as its own named
        # constant (same value) purely for readability at its call sites
        facet_row_margin = 0.45

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
        # regardless of which bin(s) the marks themselves land in.
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
            facet_font = "Arial Narrow" if narrow_font else font

            if item_labels and item_strip:
                # divided-strip layout: each item is one row spanning its
                # own threshold range. Row labels are drawn once per row,
                # like a tick label, so each item always gets its own row
                # (no packing) to keep that labelling unambiguous.
                strip_ext = (hi - lo) * 0.03

                def _draw_bounds(segs):
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
                    probe_fig = plt.figure()
                    probe_ax = probe_fig.add_subplot(111)
                    probe_text = probe_ax.text(
                        0, 0, "Ag", fontsize=labelsize, fontfamily=item_font
                    )
                    probe_fig.canvas.draw()
                    bbox = probe_text.get_window_extent(
                        renderer=probe_fig.canvas.get_renderer()
                    )
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
                # distribution_markers' item-side marks hang below the
                # persons' zero line into the top of this panel: tick
                # (1.5% of the persons panel, as _mark_len), gap, then
                # label. Row 0's strip (half-thickness 0.3 rows, as
                # bar_half below) must start clear of them, so the top
                # margin is sized from their measured depth plus
                # item_label_gap_in of padding.
                strip_margin = item_row_margin
                if distribution_markers:
                    probe_fig = plt.figure()
                    probe_ax = probe_fig.add_subplot(111)
                    probe_text = probe_ax.text(0, 0, "μ+2σ", fontsize=labelsize)
                    probe_fig.canvas.draw()
                    bbox = probe_text.get_window_extent(
                        renderer=probe_fig.canvas.get_renderer()
                    )
                    mark_label_in = (
                        bbox.height if orientation == "vertical" else bbox.width
                    ) / probe_fig.dpi
                    plt.close(probe_fig)
                    person_panel_in = (base_h if orientation == "vertical" else base_w) / 2
                    mark_depth_in = (
                        0.015 * person_panel_in
                        + mark_gap_in
                        + mark_label_in
                        + item_label_gap_in
                    )
                    strip_margin = max(
                        item_row_margin, 0.3 + mark_depth_in / row_height_in
                    )
                item_panel_in = (max_depth + 0.1 + strip_margin) * row_height_in

            elif item_labels:
                # Winsteps-style: bin items onto the histogram grid and
                # stack names as rows within each occupied bin.
                bin_idx = np.clip(
                    np.digitize(items.values, bins) - 1, 0, no_of_bins - 1
                )
                bin_centers = (bins[:-1] + bins[1:]) / 2
                item_groups = {}
                for (name, loc), b in zip(items.items(), bin_idx):
                    item_groups.setdefault(b, []).append((loc, name))

                if item_row_height is None:
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
                    mark_gap_rows = mark_gap_in / row_height_in
                    mark_text_row = -item_row_margin + mark_tick_rows + mark_gap_rows
                    mark_row_depth = max(
                        1.0,
                        mark_text_row
                        + mark_label_in / row_height_in
                        + item_label_gap_in / row_height_in
                        + 0.5,
                    )
                else:
                    mark_row_depth = (
                        -item_row_margin + item_label_gap_in / row_height_in + 0.5
                    )

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

            if facet_labels:
                # same binned-row-stacking approach as item_labels' own
                # non-strip branch above, minus the mark_row_depth
                # reservation -- distribution_markers isn't extended to
                # the facet panel, so every bin's names simply start
                # stacking from row 0
                facet_bin_idx = np.clip(
                    np.digitize(facets.values, bins) - 1, 0, no_of_bins - 1
                )
                facet_bin_centers = (bins[:-1] + bins[1:]) / 2
                facet_groups = {}
                for (name, loc), b in zip(facets.items(), facet_bin_idx):
                    facet_groups.setdefault(b, []).append((loc, name))

                if facet_row_height is None:
                    longest_facet_name = max((str(n) for n in facets.index), key=len)
                    probe_fig = plt.figure()
                    probe_ax = probe_fig.add_subplot(111)
                    probe_text = probe_ax.text(
                        0,
                        0,
                        longest_facet_name,
                        rotation=90,
                        fontsize=facet_label_size,
                        fontfamily=facet_font,
                        rotation_mode="anchor",
                    )
                    probe_fig.canvas.draw()
                    bbox = probe_text.get_window_extent(
                        renderer=probe_fig.canvas.get_renderer()
                    )
                    facet_row_height_in = bbox.height / probe_fig.dpi * 1.3
                    plt.close(probe_fig)
                else:
                    facet_row_height_in = facet_row_height

                facet_max_depth = max(
                    max((len(v) for v in facet_groups.values()), default=0), 1
                )
                # same reasoning as item_panel_in above -- the axis
                # itself later spans (facet_max_depth + 0.1 +
                # facet_row_margin) row-units, not just facet_max_depth
                facet_panel_in = (
                    facet_max_depth + 0.1 + facet_row_margin
                ) * facet_row_height_in
            else:
                facet_panel_in = None  # resolved below, once person_panel_h/w exist

            # figsize originally described the whole mirrored plot (persons
            # above, items below, sharing base_h/base_w). Now that items
            # get their own panel, persons only need the half of that
            # budget they used to occupy -- and persons' own panel always
            # gets that full share regardless of person_range, per the
            # docstring's promise that it never shrinks to make room.
            #
            # item_distribution's panel (always alongside a one-sided ax,
            # since item_distribution implies item_labels=True) is sized
            # proportionally to item_range vs person_range, so a given
            # Count/Proportion value ends up at the same physical scale
            # in both -- e.g. an item panel needing half of persons' own
            # range gets half of persons' own physical size, not a fixed
            # fraction of the whole figure regardless of what it actually
            # needs.
            #
            # The facet panel always matches items' own effective scale.
            # When items have their own panel (item_labels=True, or
            # item_distribution=True), that's the same person_panel_h/
            # person_range reference item_dist_panel_h itself uses below.
            # When persons and items still mirror onto the *same* ax
            # instead (item_labels=False), items' effective scale IS ax's
            # own: a single physical axis has one px-per-unit throughout,
            # regardless of the two sides showing different data values,
            # so ax's own combined span (person_range + item_range)
            # against its own physical size gives that same reference
            # directly -- and person_scaling then stretches the item side
            # of that span by mirror_item_scale (the facet panel picking
            # up the same factor, below).
            person_panel_h = base_h / 2 if item_labels else base_h
            person_panel_w = base_w / 2 if item_labels else base_w
            item_dist_panel_h = (
                person_panel_h * (item_range / person_range) * person_scaling
                if item_dist_panel and person_range > 0
                else (person_panel_h if item_dist_panel else 0)
            )
            item_dist_panel_w = (
                person_panel_w * (item_range / person_range) * person_scaling
                if item_dist_panel and person_range > 0
                else (person_panel_w if item_dist_panel else 0)
            )
            if facet_panel_in is None:
                if overlay_facet:
                    facet_panel_in = 0
                elif (item_labels or item_dist_panel) and person_range > 0:
                    facet_panel_in = (
                        (person_panel_h if orientation == "vertical" else person_panel_w)
                        * (facet_range / person_range)
                        * person_scaling
                        / facet_scaling
                    )
                elif person_range + item_range * mirror_item_scale > 0:
                    # mirrored map: persons and items share ax, whose total
                    # span is (person_range + item_range * mirror_item_scale)
                    # -- the item side stretched by mirror_item_scale (==
                    # person_scaling here, 1 otherwise). Size the facet's own
                    # panel so a facet count reads at the SAME physical
                    # per-count scale as an item count on that stretched item
                    # side: facet_panel_in / facet_range must equal
                    # mirror_item_scale * person_panel / total_span. (Reduces
                    # to the plain facet_range/(person_range+item_range) at
                    # mirror_item_scale=1; facet_scaling still overrides.)
                    facet_panel_in = (
                        (person_panel_h if orientation == "vertical" else person_panel_w)
                        * mirror_item_scale
                        * facet_range
                        / (person_range + item_range * mirror_item_scale)
                        / facet_scaling
                    )
                else:
                    facet_panel_in = (
                        (base_h / 3 if orientation == "vertical" else base_w / 3)
                        / facet_scaling
                    )

            # nothing needs its own panel only when there's no item panel
            # of any kind (neither a name-label panel nor a separate,
            # scale-matched distribution panel) *and* the facet
            # distribution is being overlaid onto the (then single,
            # mirrored) axis -- or, on a model with no facet structure at
            # all (SLM/PCM/RSM), there is nothing beyond persons/items to
            # share that axis with in the first place -- the two cases
            # that still collapse to a single simple axes, as in the
            # original two-family (persons/items) layout
            if not item_labels and not item_dist_panel and (overlay_facet or not has_facets):
                fig, ax = plt.subplots(figsize=(base_w, base_h))
                ax_items = None
                ax_item_dist = None
                ax_facet = None
            elif orientation == "vertical":
                margin_top_in, margin_bottom_in = 1.0, 0.6
                margin_right_in = edge_padding
                fig_w = base_w + label_margin_in + margin_right_in
                panel_sizes = [person_panel_h]
                if item_dist_panel:
                    panel_sizes.append(item_dist_panel_h)
                if item_labels:
                    panel_sizes.append(item_panel_in)
                if not overlay_facet and has_facets:
                    panel_sizes.append(facet_panel_in)
                fig_h = sum(panel_sizes) + margin_top_in + margin_bottom_in
                fig = plt.figure(figsize=(fig_w, fig_h))
                nrows = len(panel_sizes)
                gridspec_kwargs = dict(
                    height_ratios=panel_sizes,
                    hspace=0,
                    top=1 - margin_top_in / fig_h,
                    bottom=margin_bottom_in / fig_h,
                    right=1 - margin_right_in / fig_w,
                )
                if label_margin_in:
                    gridspec_kwargs["left"] = label_margin_in / fig_w
                gs = fig.add_gridspec(nrows, 1, **gridspec_kwargs)
                ax = fig.add_subplot(gs[0])
                row_i = 1
                if item_dist_panel:
                    ax_item_dist = fig.add_subplot(gs[row_i], sharex=ax)
                    row_i += 1
                else:
                    ax_item_dist = None
                if item_labels:
                    ax_items = fig.add_subplot(gs[row_i], sharex=ax)
                    row_i += 1
                else:
                    ax_items = None
                if not overlay_facet and has_facets:
                    ax_facet = fig.add_subplot(gs[row_i], sharex=ax)
                    row_i += 1
                else:
                    ax_facet = None
            else:
                margin_left_in, margin_right_in = 0.7, 0.7
                margin_top_in = 0.67
                panel_sizes = [person_panel_w]
                if item_dist_panel:
                    panel_sizes.append(item_dist_panel_w)
                if item_labels:
                    panel_sizes.append(item_panel_in)
                if not overlay_facet and has_facets:
                    panel_sizes.append(facet_panel_in)
                fig_w = sum(panel_sizes) + margin_left_in + margin_right_in
                fig_h = base_h + label_margin_in + margin_top_in
                fig = plt.figure(figsize=(fig_w, fig_h))
                ncols = len(panel_sizes)
                gridspec_kwargs = dict(
                    width_ratios=panel_sizes,
                    wspace=0,
                    left=margin_left_in / fig_w,
                    right=1 - margin_right_in / fig_w,
                    top=1 - margin_top_in / fig_h,
                )
                if label_margin_in:
                    gridspec_kwargs["bottom"] = label_margin_in / fig_h
                gs = fig.add_gridspec(1, ncols, **gridspec_kwargs)
                ax = fig.add_subplot(gs[0])
                col_i = 1
                if item_dist_panel:
                    ax_item_dist = fig.add_subplot(gs[col_i], sharey=ax)
                    col_i += 1
                else:
                    ax_item_dist = None
                if item_labels:
                    ax_items = fig.add_subplot(gs[col_i], sharey=ax)
                    col_i += 1
                else:
                    ax_items = None
                if not overlay_facet and has_facets:
                    ax_facet = fig.add_subplot(gs[col_i], sharey=ax)
                    col_i += 1
                else:
                    ax_facet = None

            # ordered outer-to-inner list of every panel beyond ax itself
            # -- the last one is the true outer edge of the whole figure
            # (gets the real Location ticks/label); everything before it
            # is an interior panel (shares the Location scale but hides
            # its own duplicate tick labels)
            extra_axes = []
            if item_dist_panel:
                extra_axes.append(ax_item_dist)
            if item_labels:
                extra_axes.append(ax_items)
            if not overlay_facet and has_facets:
                extra_axes.append(ax_facet)
            outer_ax = extra_axes[-1] if extra_axes else ax
            interior_axes = extra_axes[:-1]

            # each later panel's own opaque background would otherwise
            # paint over anything from an earlier (lower z-order) sibling
            # panel that legitimately overhangs across their shared
            # boundary -- e.g. distribution_markers' item marks, drawn
            # with clip_on=False so they aren't cut off at their own
            # panel's edge. Transparent panels let that overhang show
            # through instead of being hidden by whichever panel happens
            # to be drawn on top.
            for a in extra_axes:
                a.patch.set_visible(False)

            for a in interior_axes:
                if orientation == "vertical":
                    a.tick_params(labelbottom=False)
                else:
                    a.tick_params(labelleft=False)

            person_sign = 1 if orientation == "vertical" else -1
            item_sign = -1 if orientation == "vertical" else 1
            facet_sign = -1 if orientation == "vertical" else 1

            if item_dist_panel:
                item_ax = ax_item_dist
            elif not item_labels:
                item_ax = ax
            else:
                item_ax = None
                item_sign = None

            if overlay_facet:
                # facets overlay onto wherever items themselves are
                # drawn -- their own separate distribution panel if they
                # have one (even without item_labels, per item_dist_panel
                # above), the item_labels row panel otherwise (though
                # overlay_facet is already forced off whenever
                # item_labels=True, so this branch can't actually be
                # reached that way), or plain ax as the final fallback
                facet_dist_ax = (
                    ax_item_dist if item_dist_panel
                    else (ax_items if ax_items is not None else ax)
                )
            elif not facet_labels:
                facet_dist_ax = ax_facet
            else:
                facet_dist_ax = None

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
            # mirrored ax, sign-flipped, or the item_distribution panel,
            # unflipped), or labelled rows in a dedicated panel, or both ---
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

            if item_dist_panel:
                if orientation == "vertical":
                    ax_item_dist.spines["top"].set_visible(False)
                else:
                    ax_item_dist.spines["left"].set_visible(False)

            # --- facets: hist/kde (own panel or overlaid onto items), or
            # labelled rows in a dedicated panel ---
            # an overlaid facet shares the mirrored ax's item side, so it
            # takes the same person_scaling stretch as items; a facet in
            # its own panel does not (that panel has its own extent).
            facet_mirror_scale = mirror_item_scale if facet_dist_ax is ax else 1
            if facet_dist_ax is not None:
                if map_type == "hist":
                    facet_weights = facet_sign * facet_mirror_scale * np.ones_like(facets)
                    if prop:
                        facet_weights = facet_weights / no_of_facets

                    facet_dist_ax.hist(
                        facets,
                        weights=facet_weights,
                        bins=bins,
                        facecolor="none" if overlay_facet else facet_color,
                        edgecolor=facet_color if overlay_facet else edge_color,
                        linewidth=line_width * (2 if overlay_facet else 1),
                        hatch="///" if overlay_facet else None,
                        label=facet_label,
                        alpha=1 if overlay_facet else alpha,
                        orientation=orientation,
                    )

                else:
                    kde_facets = gaussian_kde(facets, bw_method=bw_method)(x_grid)
                    facet_curve = facet_sign * facet_mirror_scale * (
                        kde_facets if prop else kde_facets * no_of_facets
                    )

                    if overlay_facet:
                        if orientation == "vertical":
                            facet_dist_ax.plot(
                                x_grid, facet_curve, color=facet_color,
                                linewidth=line_width * 2, label=facet_label,
                            )
                        else:
                            facet_dist_ax.plot(
                                facet_curve, x_grid, color=facet_color,
                                linewidth=line_width * 2, label=facet_label,
                            )
                    elif orientation == "vertical":
                        facet_dist_ax.plot(
                            x_grid,
                            facet_curve,
                            color=facet_line_color,
                            linewidth=line_width,
                        )
                        facet_dist_ax.fill_between(
                            x_grid,
                            0,
                            facet_curve,
                            color=facet_color,
                            alpha=alpha,
                            edgecolor=facet_line_color,
                            linewidth=line_width,
                            label=facet_label,
                        )
                    else:
                        facet_dist_ax.plot(
                            facet_curve,
                            x_grid,
                            color=facet_line_color,
                            linewidth=line_width,
                        )
                        facet_dist_ax.fill_betweenx(
                            x_grid,
                            0,
                            facet_curve,
                            color=facet_color,
                            alpha=alpha,
                            edgecolor=facet_line_color,
                            linewidth=line_width,
                            label=facet_label,
                        )

            # the labelled panel is independent of whether item_ax drew a
            # hist/kde above -- item_distribution can add that panel on
            # top of either labelling style, so this is its own if/elif
            # rather than chained onto the item_ax branch above
            if item_labels and item_strip:
                boundary_margin = strip_margin
                if orientation == "vertical":
                    ax_items.set_ylim(max_depth + 0.1, -boundary_margin)
                else:
                    ax_items.set_xlim(-boundary_margin, max_depth + 0.1)

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
                else:
                    ax_items.spines["left"].set_visible(False)
                    ax_items.set_xticks([])

            elif item_labels:
                boundary_margin = item_row_margin
                if orientation == "vertical":
                    ax_items.set_ylim(max_depth + 0.1, -boundary_margin)
                else:
                    ax_items.set_xlim(-boundary_margin, max_depth + 0.1)

                for b, entries in item_groups.items():
                    x = bin_centers[b]
                    for i, (loc, name) in enumerate(sorted(entries)):
                        level = i + mark_row_depth
                        if orientation == "vertical":
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

                if orientation == "vertical":
                    ax_items.spines["top"].set_visible(False)
                    ax_items.set_yticks([])
                else:
                    ax_items.spines["left"].set_visible(False)
                    ax_items.set_xticks([])

            if facet_labels:
                if orientation == "vertical":
                    ax_facet.set_ylim(facet_max_depth + 0.1, -facet_row_margin)
                else:
                    ax_facet.set_xlim(-facet_row_margin, facet_max_depth + 0.1)

                for b, entries in facet_groups.items():
                    x = facet_bin_centers[b]
                    for i, (loc, name) in enumerate(sorted(entries)):
                        if orientation == "vertical":
                            ax_facet.text(
                                x,
                                i,
                                str(name),
                                color=facet_line_color,
                                rotation=90,
                                ha="center",
                                va="top",
                                fontsize=facet_label_size,
                                fontfamily=facet_font,
                            )
                        else:
                            ax_facet.text(
                                i,
                                x,
                                str(name),
                                color=facet_line_color,
                                ha="left",
                                va="center",
                                fontsize=facet_label_size,
                                fontfamily=facet_font,
                            )

                if orientation == "vertical":
                    ax_facet.spines["top"].set_visible(False)
                    ax_facet.set_yticks([])
                else:
                    ax_facet.spines["left"].set_visible(False)
                    ax_facet.set_xticks([])

            if distribution_markers:

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

                marker_fontsize = labelsize

                def _in_per_unit(target_ax):
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
                    gap = mark_gap_in / _in_per_unit(target_ax)
                    text_offset = sign * (mark_len + gap)

                    for label, loc in marks:
                        if orientation == "vertical":
                            target_ax.plot(
                                [loc, loc],
                                [0, sign * mark_len],
                                color=color,
                                linewidth=1,
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

                if item_mark_rows:
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
                        persons, person_sign, person_marker_color, ax, person_mark_len
                    )
                else:
                    for label, sub, fill_color, line_color in person_groups:
                        add_distribution_markers(
                            sub, person_sign, line_color, ax, person_mark_len
                        )

                if item_dist_panel:
                    add_distribution_markers(
                        items,
                        item_sign,
                        item_marker_color,
                        ax_item_dist,
                        _mark_len(ax_item_dist),
                    )
                elif item_labels and not item_strip:
                    tick_near = -item_row_margin
                    tick_far = tick_near + mark_tick_rows
                    for mark_label, mark_loc in item_mark_rows.items():
                        if orientation == "vertical":
                            ax_items.plot(
                                [mark_loc, mark_loc],
                                [tick_near, tick_far],
                                color=item_marker_color,
                                linewidth=1,
                                solid_capstyle="butt",
                                zorder=6,
                            )
                            ax_items.text(
                                mark_loc,
                                mark_text_row,
                                mark_label,
                                color=item_marker_color,
                                rotation=90,
                                ha="center",
                                va="top",
                                fontsize=marker_fontsize,
                            )
                        else:
                            ax_items.plot(
                                [tick_near, tick_far],
                                [mark_loc, mark_loc],
                                color=item_marker_color,
                                linewidth=1,
                                solid_capstyle="butt",
                                zorder=6,
                            )
                            ax_items.text(
                                mark_text_row,
                                mark_loc,
                                mark_label,
                                color=item_marker_color,
                                ha="left",
                                va="center",
                                fontsize=marker_fontsize,
                            )
                else:
                    add_distribution_markers(
                        items,
                        -person_sign if item_labels else item_sign,
                        item_marker_color,
                        ax,
                        person_mark_len,
                    )

            has_extra_panels = bool(extra_axes)

            # ax's, ax_item_dist's, and ax_facet's own Count/Proportion
            # limits are set explicitly further below (from
            # person_range/item_range/facet_range), which fully
            # supersedes pinning just one side of each to 0 here.

            padding = (hi - lo) * 0.05 if pad else 0
            loc_axis = outer_ax if (has_extra_panels and orientation == "vertical") else ax
            loc_axis_h = (
                outer_ax if (has_extra_panels and orientation == "horizontal") else ax
            )

            if has_extra_panels and orientation == "horizontal":
                outer_ax.yaxis.set_ticks_position("right")
                outer_ax.yaxis.set_label_position("right")

            def _regular_loc_ticks(axis_lo, axis_hi):
                start = np.ceil(axis_lo / tick_interval) * tick_interval
                n = int(np.floor((axis_hi - start) / tick_interval + 1e-9)) + 1
                return [start + i * tick_interval for i in range(max(n, 0))]

            if orientation == "vertical":
                ax.set_xlim(lo - padding, hi + padding)
                loc_ticks = _regular_loc_ticks(lo - padding, hi + padding)
                for a in extra_axes:
                    a.set_xticks(loc_ticks)
                loc_axis.set_xticks(loc_ticks)
            else:
                ax.set_ylim(lo - padding, hi + padding)
                loc_ticks = _regular_loc_ticks(lo - padding, hi + padding)
                for a in extra_axes:
                    a.set_yticks(loc_ticks)
                loc_axis_h.set_yticks(loc_ticks)

            has_facet_dist_panel = (
                facet_dist_ax is not None and not overlay_facet and not facet_labels
            )

            if has_extra_panels:
                if orientation == "vertical":
                    ax.spines["bottom"].set_visible(False)
                    ax.tick_params(
                        axis="x", bottom=False, top=True, labelbottom=False, labeltop=True
                    )
                else:
                    ax.spines["right"].set_visible(False)

            is_vertical = orientation == "vertical"

            # each panel's own (lo, hi) limits, set directly from
            # person_range/item_range/facet_range (an explicit *_lim
            # override, or each distribution's own natural peak,
            # resolved earlier) rather than left to independent
            # autoscale -- this is also exactly what panel sizing above
            # already assumed, so the rendered axis and the physical
            # space allocated for it always agree. ax is one-sided
            # (persons only) whenever items have their own panel
            # (item_labels row panel and/or item_distribution's scale-
            # matched one); otherwise persons and items mirror onto the
            # same ax, each of its own two sides getting its own real
            # range (the item side stretched by mirror_item_scale when
            # person_scaling rebalances them).
            if item_labels or item_dist_panel:
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

            if item_dist_panel:
                dist_lo = -item_range if item_sign < 0 else 0
                dist_hi = item_range if item_sign > 0 else 0
                if is_vertical:
                    ax_item_dist.set_ylim(dist_lo, dist_hi)
                else:
                    ax_item_dist.set_xlim(dist_lo, dist_hi)

            if has_facet_dist_panel:
                facet_lo = -facet_range if facet_sign < 0 else 0
                facet_hi = facet_range if facet_sign > 0 else 0
                if is_vertical:
                    ax_facet.set_ylim(facet_lo, facet_hi)
                else:
                    ax_facet.set_xlim(facet_lo, facet_hi)

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
                # item_dist's/facet's own zero always sits exactly at the
                # boundary shared with its neighbour -- in horizontal
                # orientation that boundary is a narrow vertical seam
                # with both panels' own "0" label sitting right next to
                # it, close enough to visually collide. Dropping the
                # secondary panel's own redundant zero (ax's own stays)
                # avoids that; not an issue in vertical orientation,
                # where the seam is horizontal and the two labels don't
                # compete for the same space.
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

            if item_labels or item_dist_panel:
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

            if item_dist_panel:
                if item_sign > 0:
                    _relabel(ax_item_dist, item_range, item_step, 0, 1, drop_zero=True)
                else:
                    _relabel(ax_item_dist, 0, 1, item_range, item_step, drop_zero=True)

            if has_facet_dist_panel:
                if facet_sign > 0:
                    _relabel(ax_facet, facet_range, facet_step, 0, 1, drop_zero=True)
                else:
                    _relabel(ax_facet, 0, 1, facet_range, facet_step, drop_zero=True)

            # the explicit axhline/axvline boundary lines and the "hide
            # the gridline that would otherwise sit right under them"
            # cleanup both need the *final* tick set from _relabel above
            # -- doing this earlier (against whatever ticks autoscale
            # had chosen before the real limits were even set) hid
            # whichever gridline happened to occupy that position in the
            # stale tick list, not necessarily the one actually at 0
            if is_vertical:
                ax.axhline(0, color="black", linewidth=1.3, zorder=5)
                for tick, gridline in zip(ax.get_yticks(), ax.yaxis.get_gridlines()):
                    if tick == 0:
                        gridline.set_visible(False)
                if has_facet_dist_panel:
                    ax_facet.axhline(0, color="black", linewidth=1.3, zorder=5)
                    for tick, gridline in zip(
                        ax_facet.get_yticks(), ax_facet.yaxis.get_gridlines()
                    ):
                        if tick == 0:
                            gridline.set_visible(False)
            else:
                ax.axvline(0, color="black", linewidth=1.3, zorder=5)
                if has_facet_dist_panel:
                    ax_facet.axvline(0, color="black", linewidth=1.3, zorder=5)
                    for tick, gridline in zip(
                        ax_facet.get_xticks(), ax_facet.xaxis.get_gridlines()
                    ):
                        if tick == 0:
                            gridline.set_visible(False)
                for tick, gridline in zip(ax.get_xticks(), ax.xaxis.get_gridlines()):
                    if tick == 0:
                        gridline.set_visible(False)

            if is_vertical:
                loc_axis.set_xlabel(
                    "Location", fontsize=axis_font_size, fontweight="bold"
                )
                ax.set_ylabel(
                    ("Density" if map_type == "kde" else "Proportion") if prop else "Count",
                    fontsize=axis_font_size,
                    fontweight="bold",
                )
                if item_distribution:
                    # one shared "Count"/"Density"/"Proportion" label instead
                    # of one per panel -- re-centre ax's own label (rather than also
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
                    # correctly into set_label_coords. Only recentres
                    # across ax+ax_item_dist -- doesn't additionally
                    # account for a facet panel also being present, so on
                    # a facet-element model with both item_distribution and
                    # a non-overlaid facet panel the label centres on the
                    # persons+items span, not the full figure.
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
                if facet_labels:
                    ax_facet.set_ylabel(
                        facet_label, fontsize=axis_font_size, fontweight="bold"
                    )
            else:
                loc_axis_h.set_ylabel(
                    "Location", fontsize=axis_font_size, fontweight="bold"
                )
                ax.set_xlabel(
                    ("Density" if map_type == "kde" else "Proportion") if prop else "Count",
                    fontsize=axis_font_size,
                    fontweight="bold",
                )
                if item_distribution:
                    # one shared "Count"/"Density"/"Proportion" label
                    # instead of one per panel -- re-centre ax's own label across both panels'
                    # combined span (in ax's own axes-fraction coordinates,
                    # since item_dist sits directly to its right) instead
                    # of also labelling ax_item_dist. Measured the same way
                    # as the vertical case above, for the same reason:
                    # label.get_position() after a draw returns a display-
                    # space value from matplotlib's internal auto-layout
                    # transform, not one set_label_coords can reuse. Same
                    # facet-panel caveat as the vertical case above.
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
                if facet_labels:
                    ax_facet.set_xlabel(
                        facet_label, fontsize=axis_font_size, fontweight="bold"
                    )

            ax.tick_params(axis="x", labelsize=labelsize)
            ax.tick_params(axis="y", labelsize=labelsize)
            for a in extra_axes:
                a.tick_params(axis="x", labelsize=labelsize)
                a.tick_params(axis="y", labelsize=labelsize)

            # A facet-label panel directly after the item-label panel needs
            # no dividing line, and only a small gap: measured from the
            # item panel's outermost content (strips, leader-line labels,
            # names) to the nearest facet label, anything beyond
            # facet_gap_in is closed up by moving the facet panel in and
            # shrinking the figure by the same amount, so no row or strip
            # changes size.
            if ax_items is not None and ax_facet is not None and facet_labels:
                facet_gap_in = 2 * item_label_gap_in
                renderer = fig.canvas.get_renderer()
                item_boxes = [
                    a.get_window_extent(renderer)
                    for a in list(ax_items.patches) + list(ax_items.texts) + list(ax_items.lines)
                    if a.get_visible()
                ]
                facet_boxes = [
                    t.get_window_extent(renderer)
                    for t in ax_facet.texts
                    if t.get_visible() and t.get_text().strip()
                ]
                fig_w_in, fig_h_in = fig.get_size_inches()
                if is_vertical:
                    ax_items.spines["bottom"].set_visible(False)
                else:
                    ax_items.spines["right"].set_visible(False)
                if item_boxes and facet_boxes:
                    if is_vertical:
                        gap_px = min(b.y0 for b in item_boxes) - max(b.y1 for b in facet_boxes)
                    else:
                        gap_px = min(b.x0 for b in facet_boxes) - max(b.x1 for b in item_boxes)
                    excess_in = gap_px / fig.dpi - facet_gap_in
                    if excess_in > 0:
                        facet_pos = ax_facet.get_position()
                        after_seam = [
                            a for a in fig.axes
                            if (a.get_position().y1 <= facet_pos.y1 + 1e-9 if is_vertical
                                else a.get_position().x0 >= facet_pos.x0 - 1e-9)
                        ]
                        new_size = (
                            (fig_w_in, fig_h_in - excess_in) if is_vertical
                            else (fig_w_in - excess_in, fig_h_in)
                        )
                        for a in fig.axes:
                            p = a.get_position()
                            if is_vertical:
                                y0_in = p.y0 * fig_h_in - (0 if a in after_seam else excess_in)
                                a.set_position([
                                    p.x0, y0_in / new_size[1],
                                    p.width, p.height * fig_h_in / new_size[1],
                                ])
                            else:
                                x0_in = p.x0 * fig_w_in - (excess_in if a in after_seam else 0)
                                a.set_position([
                                    x0_in / new_size[0], p.y0,
                                    p.width * fig_w_in / new_size[0], p.height,
                                ])
                        fig.set_size_inches(*new_size)
                        fig_w, fig_h = new_size

            if title is not None:
                if has_extra_panels:
                    title_y = 1 - (margin_top_in * 0.4) / fig_h
                    fig.suptitle(
                        title, fontsize=title_font_size, fontweight="bold", y=title_y
                    )
                else:
                    ax.set_title(title, fontsize=title_font_size, fontweight="bold")

            legend_handles, legend_labels = ax.get_legend_handles_labels()
            if item_dist_panel:
                dist_handles, dist_labels = ax_item_dist.get_legend_handles_labels()
                legend_handles += dist_handles
                legend_labels += dist_labels
            if facet_dist_ax is not None and not overlay_facet and not facet_labels:
                fdist_handles, fdist_labels = ax_facet.get_legend_handles_labels()
                legend_handles += fdist_handles
                legend_labels += fdist_labels
            legend_loc = "upper left" if person_sign < 0 else "upper right"
            ax.legend(legend_handles, legend_labels, loc=legend_loc)

            # Stretch panel(s) -- bolted onto the right of the whole
            # figure, after everything else is built and sized. omega
            # isn't a logit value (see the comment by stretch_omega's own
            # definition above), so it gets its own axis/scale entirely,
            # centred on 1 rather than 0, one bar per rater in plain
            # facet-name order (unrelated to whatever order the shift
            # panel's own binning happened to place them in). bistretch
            # gets two such panels, stacked (omega_items above
            # omega_thresholds), sharing the same rater order/x-axis.
            if stretch_omega is not None and show_stretch:
                stretch_main_axes = [ax] + extra_axes
                stretch_orig_w, stretch_orig_h = fig.get_size_inches()

                stretch_gap_in = 0.073
                stretch_margin_in = 0.55
                n_stretch = len(stretch_omega)
                raters_order = list(next(iter(stretch_omega.values())).index)
                n_raters = len(raters_order)
                stretch_col_w_in = max(1.8, 0.35 * n_raters)

                stretch_extra_w_in = stretch_gap_in + stretch_col_w_in + stretch_margin_in
                stretch_new_w = stretch_orig_w + stretch_extra_w_in
                fig.set_size_inches(stretch_new_w, stretch_orig_h)
                stretch_shrink = stretch_orig_w / stretch_new_w
                for a in stretch_main_axes:
                    pos = a.get_position()
                    a.set_position(
                        [pos.x0 * stretch_shrink, pos.y0, pos.width * stretch_shrink, pos.height]
                    )

                stretch_tops = [a.get_position().y1 for a in stretch_main_axes]
                stretch_bots = [a.get_position().y0 for a in stretch_main_axes]
                stretch_y0, stretch_y1 = min(stretch_bots), max(stretch_tops)
                x0_frac = 1 - (stretch_col_w_in + stretch_margin_in) / stretch_new_w
                w_frac = stretch_col_w_in / stretch_new_w

                expand_color = "gray" if black else "steelblue"
                compress_color = "darkgray" if black else "indianred"
                omega_symbols = {
                    "omega": "$\\omega$",
                    "omega_items": "$\\omega_{items}$",
                    "omega_thresholds": "$\\omega_{thresholds}$",
                }

                row_gap_frac = 0.06 * (stretch_y1 - stretch_y0)
                row_h = (stretch_y1 - stretch_y0 - row_gap_frac * (n_stretch - 1)) / n_stretch

                # Shared y-limits across every stretch panel (e.g.
                # bistretch's omega_items/omega_thresholds), so a bar's
                # length is comparable panel to panel rather than each
                # auto-scaling to its own data independently.
                all_omega_vals = [
                    v for s in stretch_omega.values() for v in s.values if v == v
                ]
                if all_omega_vals:
                    om_lo, om_hi = min(all_omega_vals), max(all_omega_vals)
                    om_pad = max((om_hi - om_lo) * 0.15, 0.05)
                    shared_ylim = (min(om_lo, 1) - om_pad, max(om_hi, 1) + om_pad)
                else:
                    shared_ylim = (0.5, 1.5)

                for i, (omega_key, omega_series) in enumerate(stretch_omega.items()):
                    row_top = stretch_y1 - i * (row_h + row_gap_frac)
                    ax_stretch = fig.add_axes([x0_frac, row_top - row_h, w_frac, row_h])
                    bar_xs = list(range(n_raters))
                    bar_vals = [omega_series.get(r, np.nan) for r in raters_order]
                    bar_colors = [
                        (compress_color if v < 1 else expand_color)
                        if v == v else "lightgray"
                        for v in bar_vals
                    ]
                    ax_stretch.bar(
                        bar_xs,
                        [(v - 1) if v == v else 0 for v in bar_vals],
                        bottom=1,
                        color=bar_colors,
                        edgecolor=edge_color,
                        width=0.6,
                    )
                    ax_stretch.axhline(1, color=edge_color, linewidth=1)
                    ax_stretch.set_xlim(-0.6, n_raters - 0.4)
                    ax_stretch.set_ylim(shared_ylim)
                    ax_stretch.set_xticks(bar_xs)
                    ax_stretch.set_ylabel(
                        omega_symbols.get(omega_key, omega_key),
                        fontsize=axis_font_size,
                        fontweight="bold",
                    )
                    ax_stretch.yaxis.tick_right()
                    ax_stretch.yaxis.set_label_position("right")
                    ax_stretch.spines["top"].set_visible(False)
                    ax_stretch.spines["left"].set_visible(False)
                    ax_stretch.tick_params(axis="y", labelsize=labelsize)
                    if i == n_stretch - 1:
                        ax_stretch.set_xticklabels(
                            raters_order,
                            fontsize=facet_label_size,
                            fontfamily=facet_font,
                            rotation=90,
                        )
                    else:
                        ax_stretch.tick_params(labelbottom=False)

            # Thin count-axis tick labels that would otherwise collide
            # (e.g. a small distribution squeezed onto a short stretch of
            # axis). Measured against the final layout, after every panel
            # resize above: on each side of zero, keep every k-th tick,
            # with k the smallest value whose on-screen spacing fits the
            # labels.
            def _thin_count_ticks(ax_obj):
                axis = ax_obj.yaxis if is_vertical else ax_obj.xaxis
                positions = list(axis.get_majorticklocs())
                if len(positions) < 3:
                    return
                formatter = axis.get_major_formatter()
                labels = [formatter(p, i) for i, p in enumerate(positions)]
                to_px = ax_obj.transData.transform

                def _px(v):
                    return to_px((0, v))[1] if is_vertical else to_px((v, 0))[0]

                pt_to_px = fig.dpi / 72
                if is_vertical:
                    min_gap = labelsize * 1.3 * pt_to_px
                else:
                    widest = max(
                        TextPath((0, 0), lab, size=labelsize).get_extents().width
                        for lab in labels
                    )
                    min_gap = (widest + 0.6 * labelsize) * pt_to_px

                kept = [(p, lab) for p, lab in zip(positions, labels) if p == 0]
                thinned = False
                for side in (1, -1):
                    ticks = sorted(
                        (pl for pl in zip(positions, labels) if pl[0] * side > 0),
                        key=lambda pl: abs(pl[0]),
                    )
                    # k = len(ticks) + 1 keeps none of this side's ticks
                    # (just the shared zero): used when even a lone tick
                    # would sit too close to zero to be legible
                    k = 1
                    while k <= len(ticks):
                        chosen = [0.0] + [ticks[i][0] for i in range(k - 1, len(ticks), k)]
                        gaps = [abs(_px(a) - _px(b)) for a, b in zip(chosen, chosen[1:])]
                        if min(gaps) >= min_gap:
                            break
                        k += 1
                    thinned |= k > 1
                    kept += [ticks[i] for i in range(k - 1, len(ticks), k)]
                if not thinned:
                    return

                zero_hidden = any(
                    t == 0 and not g.get_visible()
                    for t, g in zip(positions, axis.get_gridlines())
                )
                kept.sort()
                axis.set_ticks([p for p, _ in kept])
                axis.set_ticklabels([lab for _, lab in kept], fontsize=labelsize)
                if zero_hidden:
                    for t, g in zip(axis.get_majorticklocs(), axis.get_gridlines()):
                        if t == 0:
                            g.set_visible(False)

            _thin_count_ticks(ax)
            if item_dist_panel:
                _thin_count_ticks(ax_item_dist)
            if has_facet_dist_panel:
                _thin_count_ticks(ax_facet)

            if filename is not None:
                fig.savefig(filename + f".{file_format}", dpi=dpi)

            plt.close(fig)

        return fig
