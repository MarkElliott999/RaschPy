import colorsys
import re
import copy
import warnings
from contextlib import contextmanager
from raschpy import _progress
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


def _f_reference(W, q, n, ncp=0.0):
    """
    Upper-tail p-value of a Wald statistic W = d' V^+ d on q degrees of freedom whose covariance V is estimated
    from the responses of n independent persons, from the F reference of Hotelling's T^2:
    F = W (n - q) / (q (n - 1)) on (q, n - q), noncentral with noncentrality ncp when ncp > 0. For a comparison
    of independent groups whose covariances may differ (V = V_1 + V_2), n = nu + 1 with nu the multivariate Welch
    degrees of freedom of _welch_df. Returns 1 when n <= q (V cannot be estimated with q
    dimensions from n persons).

    The reference is exact only when the per-person contributions to d are multivariate normal and V is their
    sample covariance (Hotelling's T^2). Here d is a function of estimates and V a sandwich (or bootstrap)
    covariance, so the F reference is an approximation, not an exact test: it allows for the uncertainty of
    estimating V from n persons, which a chi-square reference ignores (chi-square is its limit as n grows), but
    the per-person contributions are neither normal nor exactly averaged.
    """
    W, q, n = float(W), int(q), float(n)
    if q < 1:
        return np.nan
    d2 = n - q
    if d2 <= 0:
        return 1.0
    F = W * d2 / (q * (n - 1))
    if ncp > 0:
        from scipy.stats import ncf

        return float(ncf.sf(F, q, d2, ncp))
    return float(f_dist.sf(F, q, d2))


def _welch_df(covs, ns):
    """
    Nel & van der Merwe (1986) degrees of freedom nu for a Wald statistic comparing independent groups whose
    estimates' covariances covs (each V_g, already the covariance of group g's estimates) come from n_g persons:
    nu = [tr(S^2) + tr(S)^2] / sum_g [tr(V_g^2) + tr(V_g)^2] / (n_g - 1), S = sum_g V_g, on the support of S.
    The F reference is then _f_reference(W, q, nu + 1), i.e. F = W (nu - q + 1) / (q nu) on (q, nu - q + 1).
    With one parameter it is the Welch-Satterthwaite df; with equal covariances and sizes it is n_1 + n_2 - 2.
    """
    S = sum(covs)
    w, Q = np.linalg.eigh((S + S.T) / 2)
    P = Q[:, w > 1e-10 * w.max()]
    S = P.T @ S @ P
    den = sum(
        (np.trace(G @ G) + np.trace(G) ** 2) / (n - 1)
        for G, n in ((P.T @ V @ P, n) for V, n in zip(covs, ns))
    )
    return float((np.trace(S @ S) + np.trace(S) ** 2) / den)


def _esf_lcw(b):
    """_esf.log_category_weights (imported when first needed)."""
    from raschpy import _esf

    return _esf.log_category_weights(b)


# Apply base style once at module load rather than on every plot call.
# Individual methods can override via plt.rc_context() if needed.
plt.style.use("seaborn-v0_8-white")
sns.set_style("whitegrid")


def _disorder_flag(widths):
    """category_stats Disordered: True for a negative category width, False for a positive one, NaN for one the data
    do not identify (inf / NaN, next to a category kept with unused_categories='null')."""
    w = np.asarray(widths, dtype=float)
    if np.isfinite(w).all():
        return w < 0
    out = np.empty(len(w), dtype=object)
    for k, v in enumerate(w):
        out[k] = bool(v < 0) if np.isfinite(v) else np.nan
    return out


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
        sim's real instance attributes, not its __getattr__-resolved
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
            if issubclass(
                category, (DeprecationWarning, PendingDeprecationWarning, FutureWarning)
            ):
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


COND_VAR_ZERO = 1e-10  # conditional variances below this are 0 (responses fixed by the total)
MARTIN_LOF_SPARSE = 0.25  # share of score cells with < 5 persons above which the chi-square p is conservative


_FIT_FIRST = (
    "person", "person_estimates", "csem", "score_lookup", "score_lookup_table", "icc", "crcs",
    "threshold_ccs", "iic", "tcc", "test_info", "test_csem", "wright_map",
)
_PERSONS_FIRST = ("class_intervals", "class_intervals_cats", "class_intervals_thresholds")
_FIT_STATISTICS_FIRST = ("std_residuals_plot",)


def _calibrate_on_demand(cls, fitted):
    """
    Wrap cls's results and plotting methods so that, like fit_statistics() and the stats tables,
    they calibrate an uncalibrated object first (calibrate() with its defaults); the class-interval
    methods also estimate persons, and std_residuals_plot() also computes the fit statistics, when
    these are missing. fitted: the attribute a calibration sets.
    """
    import functools

    def wrap(name, f):
        @functools.wraps(f)
        def method(self, *args, **kwargs):
            if not hasattr(self, fitted):
                self.calibrate()
            if name in _PERSONS_FIRST and not hasattr(self, "persons"):
                self.person_estimates()
            if name in _FIT_STATISTICS_FIRST and not hasattr(self, "std_residual_df"):
                self.fit_statistics()
            return f(self, *args, **kwargs)

        return method

    for name in _FIT_FIRST + _PERSONS_FIRST + _FIT_STATISTICS_FIRST:
        f = getattr(cls, name, None)
        if f is not None:
            setattr(cls, name, wrap(name, f))


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
    # Purification variance: redo the anchor selection in every bootstrap resample (or draw from the
    # estimates' sampling distribution), so a tested unit's SE includes the variability of choosing the anchor set
    # from the data. The draws are first shifted so that the tested unit and the anchors kept on the observed data
    # have no group difference (the null hypothesis for the tested unit); units the observed selection dropped keep
    # their observed differences. False keeps the anchor set selected on the observed data fixed (v1.4 behaviour).
    _RESELECT_ANCHORS = True
    _RESELECT_DRAWS = 2000  # draws from N(estimates, covariance) when there are no bootstrap resamples

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
    # the 'colorblind multi' palette's black entry (Okabe & Ito's
    # 8th colour) would be invisible or near-invisible against it. Curve-
    # colouring code checks this set and substitutes white for that one
    # entry when the active plot_style is a member -- see palette_dict
    # construction in each model class's plot_data(). Deliberately
    # excludes 'dark' (seaborn darkgrid's light-grey background, against
    # which black remains perfectly legible).
    _DARK_BACKGROUND_STYLES = {
        "black",
        "solarized-dark",
        "slate",
        "blueprint",
        "chalkboard",
    }

    def _apply_plot_style(self, plot_style):
        """
        Apply a named plot_style scheme via seaborn/matplotlib rcParams.

        Shared by every plot_data()/wright_map()/plot_anchor_selection()/
        std_residuals_hist() call across all four model classes, so a new
        scheme only needs to be added once, here, to become available
        everywhere. Falls back to 'white' for an unrecognised name rather
        than raising, matching the original if/else's silent
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
        subclass's calibrate() uses (via self._build_pairwise_matrix(),
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
        # the model's matrix uses (one row per item, or — for PCM —
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
        directionally_isolated_report = sorted(
            set(row_items[unexplained_zero].tolist())
        )
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
            n_rest, _ = connected_components(
                directed[np.ix_(keep, keep)], directed=True, connection="strong"
            )
            scc_explained = n_rest == 1

        if is_connected:
            if not strongly_connected and not scc_explained:
                scc_groups = {}
                for node_idx, label in enumerate(scc_labels):
                    scc_groups.setdefault(label, set()).add(str(row_items[node_idx]))
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
                f"identified only up to its origin -- and un-smoothed "
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
                         the non-iterative methods below still return an answer.
            'cos'      — cosine maximisation method (Kou & Lin, 2014): sum of
                         L2-normalised columns of the reciprocal matrix. Fast,
                         robust, non-iterative; least sensitive to heavy-tailed
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
            # Warm-start from the non-iterative cosine-maximisation solution,
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
                np.add(weights[:, np.newaxis], weights[np.newaxis, :], out=weight_pairs)
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
    # items) use the information-weighted estimator under threshold_method='cpat-cl'.
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
            log_ratio = np.log(np.where(finite, a, 1.0)) - np.log(
                np.where(finite, b, 1.0)
            )
            weight = np.where(finite, 2.0 * a * b / (a + b), 0.0)
        total = weight.sum()
        if total == 0:
            return np.nan
        return float((weight * (log_ratio + loc)).sum() / total)

    @staticmethod
    def _cpat_cl_distance(num, den, loc, constant):
        """
        CPAT threshold distance maximising the composite likelihood of the threshold patterns over item pairs
        (threshold_method='cpat-cl'; not the conditional maximum likelihood of estimation='cml').

        Given that a person's responses to items i != j are (k, k) or
        (k-1, k+1), theta cancels and
        logit P(k, k) = D - (delta_i - delta_j), with D = tau_{k+1} - tau_k,
        so num_ij ~ Binomial(num_ij + den_ij, p_ij) with the item differences
        as a known offset. D maximises the resulting binomial likelihood
        (strictly concave), found by Newton's method safeguarded by a bracket
        on the root of the score (which is strictly decreasing in D), with
        bisection whenever a step leaves the bracket. Unlike the
        information-weighted estimator, the weights are the model's
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

        Computed in symmetric form: with B^+ = S S' (S = Q w^(1/2) from its
        eigendecomposition), I - B_n B^+ = T (I - S' B_n S) S' (T = Q w^(-1/2),
        so S' T = I), so (I - H_n)^(-1/2) u = T (I - S' B_n S)^(-1/2) S' u, from
        a symmetric eigendecomposition. (The direct eigendecomposition of the
        non-symmetric I - H_n can give a singular eigenvector matrix when B_n
        has low rank, which left the correction out for a whole chunk of
        persons.) The part of u outside the support of B^+ is unchanged.
        """
        U = np.asarray(U, dtype=float)
        N, p = U.shape
        A = np.asarray(bread_inv, dtype=float)
        w, Q = np.linalg.eigh((A + A.T) / 2)
        keep = w > 1e-10 * max(w.max(), 1e-300)
        Qk = Q[:, keep]
        S, T = Qk * np.sqrt(w[keep]), Qk / np.sqrt(w[keep])
        out = U.copy()
        eye = np.eye(S.shape[1])
        for start in range(0, N, chunk):
            sl = slice(start, min(start + chunk, N))
            M = eye[None] - S.T @ person_bread(sl) @ S  # S' B_n S for each person (batched matmul)
            lam, V = np.linalg.eigh((M + M.transpose(0, 2, 1)) / 2)
            good = np.isfinite(lam).all(axis=1) & (lam.min(axis=1) > 1e-8)
            if not good.any():
                continue
            u = U[sl][good]
            y = u @ S  # S' u
            Vg = V[good]  # z = V diag(lam^-1/2) V' y for each person
            z = (Vg @ ((Vg.transpose(0, 2, 1) @ y[:, :, None])[:, :, 0] / np.sqrt(lam[good]))[:, :, None])[:, :, 0]
            adj = z @ T.T + (u - (u @ Qk) @ Qk.T)
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
            new = np.where(
                finite, new, np.where(bracketed, 0.5 * (lo[a] + hi[a]), t - np.sign(t))
            )
            done = finite & (
                (np.abs(new - t) < tolerance)
                | (bracketed & (hi[a] - lo[a] < tolerance))
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
        group's item estimates, observed = the focal group's).

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

    @staticmethod
    def _anchor_wald_stats(d, Va, Vo, keep, n_a, n_o):
        """
        The anchor-selection Wald statistics, vectorised over draws (rows): d (draws x candidates) holds
        anchor - observed for each candidate, Va and Vo the covariance matrices of the anchor and observed
        values (Va zero for fixed anchors), keep (draws x candidates) the current anchor set.

        tc is the precision-weighted mean of d over the kept candidates, weights w_j = 1 / (Va_jj + Vo_jj).
        Each candidate's deviation d_i - tc has variance c'Va c + c'Vo c, c = e_i - a, a the weights
        normalised over the kept candidates (for a kept candidate with diagonal V: v_i (1 - w_i / sum(w))).
        t = deviation / SE; df = Welch-Satterthwaite from the two components and the persons n_a, n_o
        behind them (n_o - 1 when the anchors are fixed; infinite, i.e. a normal reference, when n_o is None).

        Returns tc (draws,), dev, se, t, df, p (each draws x candidates).
        """
        d = np.atleast_2d(d)
        keep = np.atleast_2d(keep)
        w = 1.0 / (np.diag(Va) + np.diag(Vo))
        a = w * keep
        a = a / a.sum(1, keepdims=True)
        tc = (a * d).sum(1)
        dev = d - tc[:, None]

        def part(V):
            aV = a @ V
            return np.clip(np.diag(V)[None, :] - 2.0 * aV + (aV * a).sum(1)[:, None], 0.0, None)

        vo = part(Vo)
        va = part(Va) if np.any(Va) else np.zeros_like(vo)
        var = va + vo
        pos = var > 0
        se = np.sqrt(var)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(pos, dev / np.where(pos, se, 1.0), 0.0)
            if n_o is None:
                df = np.full_like(var, np.inf)
            elif n_a is None or not np.any(Va):
                df = np.full_like(var, max(n_o - 1, 1))
            else:
                df = np.where(pos, var**2 / (va**2 / max(n_a - 1, 1) + vo**2 / max(n_o - 1, 1)), np.inf)
        p = 2 * t_dist.sf(np.abs(t), df)
        return tc, dev, se, t, df, p

    def _wald_anchor_selection(self, anchor, observed, se, alpha=0.05, min_anchors=6, anchor_cov=None,
                               observed_cov=None, n_anchor=None, n_observed=None):
        """
        Wald-test-based anchor-item selection: an alternative to
        _robust_anchor_selection() for deciding which supplied anchor
        items are safe to use when computing the translation constant.

        For each candidate anchor item, tests H0: this item's deviation
        from the current (precision-weighted) translation line is zero,
        t_i = (anchor_i - (observed_i + tc)) / SE(deviation_i), referred
        to a t-distribution. The deviation's variance counts the
        observed value's error, the anchor value's error when the anchor
        values are themselves estimates (anchor_cov: e.g. the reference
        group's calibration in DIF purification), and the item's own
        share of tc (see _anchor_wald_stats). Fixed anchor values
        (anchor_cov=None, as in calibrate_anchor) carry no error.

        Uses sequential exclusion (recompute tc and re-test on the
        current remaining set at each step, dropping the largest |t|),
        not a single pass against a translation constant fixed from the
        full candidate set: tc is a precision-weighted mean, so a single
        severely off anchor item can drag it far enough that other,
        genuinely fine anchor items also fail the test (collateral
        flagging; "artificial DIF" in Andrich's sense).

        Parameters
        ----------
        anchor : pandas.Series
            Reference values, indexed by item name.
        observed : pandas.Series
            Freshly-calibrated values for the same items, indexed by item
            name. Only items present in both anchor and observed are used.
        se : pandas.Series
            SE of `observed`, indexed by item name. Used for the
            observed values' variances when observed_cov is None.
        alpha : float, default 0.05
            Significance level for each item's test.
        min_anchors : int, default 6
            Floor on the number of surviving items: skip trimming
            entirely (with a warning) if the initial common set is at or
            below this floor; stop early (also with a warning) if
            trimming reaches this floor before every remaining item
            clears alpha.
        anchor_cov, observed_cov : pandas.DataFrame or None
            Covariance matrices of the anchor and observed values
            (indexed by item name both ways). None for anchor_cov: fixed
            anchor values; None for observed_cov: squared SEs on the
            diagonal.
        n_anchor, n_observed : int or None
            Persons behind the anchor and observed values, for the
            degrees of freedom: Welch-Satterthwaite when the anchor
            values are estimates, n_observed - 1 when they are fixed. A
            normal reference if n_observed is None.

        Returns
        -------
        dict — same shape as _robust_anchor_selection(), so callers can
        use either interchangeably:
            table : pandas.DataFrame
                Anchor, Observed, SE, Deviation, SE (deviation), t, df,
                p, Selected — indexed by item name. A dropped item's
                Deviation to p are from the step at which it was dropped.
            selected_anchors, dropped_anchors : pandas.Index
            original_anchor_corr, original_anchor_sd_ratio : float
                Descriptive only (Pearson corr / SD ratio before
                trimming), kept for comparability with
                _robust_anchor_selection.
            anchor_corr, anchor_sd_ratio : float
                Same, after trimming.
            tc : float
                Precision-weighted translation constant over the
                finally-selected items, weights 1 / (anchor variance +
                observed variance).
            weights : pandas.Series
                Those weights, for every common item.
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
        n = len(common)
        if observed_cov is not None and not set(common).issubset(observed_cov.index):
            observed_cov = None
        Vo = (observed_cov.loc[common, common].to_numpy(float) if observed_cov is not None
              else np.diag(table["SE"].to_numpy(float) ** 2))
        if anchor_cov is not None and not set(common).issubset(anchor_cov.index):
            raise ValueError("anchor_cov does not cover every common item.")
        Va = anchor_cov.loc[common, common].to_numpy(float) if anchor_cov is not None else np.zeros((n, n))
        d = (table["Anchor"] - table["Observed"]).to_numpy(float)
        weights = pd.Series(1.0 / (np.diag(Va) + np.diag(Vo)), index=common)

        original_anchor_corr = table["Anchor"].corr(table["Observed"])
        original_anchor_sd_ratio = table["Observed"].std() / table["Anchor"].std()

        def _wald(keep):
            tc, dev, s, t, df, p = self._anchor_wald_stats(d, Va, Vo, keep, n_anchor, n_observed)
            return float(tc[0]), dev[0], s[0], t[0], df[0], p[0]

        stat_cols = ["Deviation", "SE (deviation)", "t", "df", "p"]
        keep = np.ones(n, bool)
        if n <= min_anchors:
            warnings.warn(
                f"Only {n} items have both an anchor and an "
                f"observed value, at or below min_anchors={min_anchors}. "
                f"Skipping Wald-based outlier trimming; all supplied anchor "
                f"items will be used.",
                UserWarning,
                stacklevel=2,
            )
            tc, dev, s, t, df, p = _wald(keep)
            for col, v in zip(stat_cols, (dev, s, t, df, p)):
                table[col] = v
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
                "weights": weights,
                "converged": False,
            }

        dropped_anchors = []
        dropped_stats = {}  # position -> statistics at its removal step
        converged = True

        while True:
            tc, dev, s, t, df, p = _wald(keep)
            if (p[keep] > alpha).all():
                break
            if keep.sum() <= min_anchors:
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
            worst = int(np.argmax(np.where(keep, np.abs(t), -1.0)))
            dropped_anchors.append(common[worst])
            dropped_stats[worst] = (dev[worst], s[worst], t[worst], df[worst], p[worst])
            keep[worst] = False

        for col in stat_cols:
            table[col] = np.nan
        for col, v in zip(stat_cols, (dev, s, t, df, p)):
            table.loc[keep, col] = v[keep]
        for pos, vals in dropped_stats.items():
            table.iloc[pos, [table.columns.get_loc(c) for c in stat_cols]] = vals
        selected_anchors = common[keep]
        table["Selected"] = keep

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
            "weights": weights,
            "converged": converged,
        }

    def _location_covariance(self):
        """
        Covariance matrix (pandas.DataFrame, indexed by item name both ways) of the item locations (PCM: central
        locations) from the current standard errors: the analytic covariance for se_method 'sandwich'/'hessian',
        else the covariance of the stored bootstrap resamples (rows with a non-finite value dropped), else None.
        """
        if getattr(self, "se_method", None) in ("sandwich", "hessian"):
            if hasattr(self, "_central_covariance"):
                return self._central_covariance()
            cov = getattr(self, "se_covariance", None)
            return cov["items"] if isinstance(cov, dict) else cov
        boot = getattr(self, "central_bootstrap" if hasattr(self, "_central_covariance") else "item_bootstrap", None)
        return self._boot_cov(boot)

    @staticmethod
    def _boot_cov(boot, cols=None):
        """Covariance (DataFrame) of bootstrap resamples boot[cols], rows with a non-finite value dropped; None if
        boot is None, lacks a column, or keeps fewer than 3 rows."""
        if boot is None:
            return None
        cols = list(boot.columns) if cols is None else list(cols)
        if not set(cols).issubset(boot.columns):
            return None
        X = boot[cols].to_numpy(float)
        X = X[np.isfinite(X).all(1)]
        if len(X) < 3:
            return None
        return pd.DataFrame(np.atleast_2d(np.cov(X.T)), index=cols, columns=cols)


    def _bootstrap_value_matrix(self, samples, attr, index):
        """
        (B, len(index)) pandas.DataFrame of a Series-valued attribute
        (e.g. items, facet_effects_global, lambda_centrality) across a
        model's stored bootstrap resamples, columns=index. Samples
        whose attribute doesn't cover the full index (a resample that
        dropped a unit, or where calibration produced a ragged result)
        are skipped rather than filled with NaN, matching
        _store_rater_se's validity filter.
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

    @staticmethod
    def _emit_loo_warnings(loo_caught, full_caught, n_units):
        """
        Re-emit the warnings of _purification_loo_tc's anchor selections: the
        whole-set selection's as raised, and each distinct per-unit
        (leave-one-out) warning once, with the number of units it occurred
        for, unless the whole-set selection gave the same warning.
        """

        def key(w):
            return (w.category, re.sub(r"\d+", "#", str(w.message)))

        full_keys = {key(w) for w in full_caught}
        for w in full_caught:
            warnings.warn(w.message, w.category, stacklevel=4)
        seen = {}
        for w in loo_caught:
            k = key(w)
            if k in full_keys:
                continue
            if k not in seen:
                seen[k] = [w, 0]
            seen[k][1] += 1
        for w, count in seen.values():
            msg = str(w.message)
            if count > 1 or n_units > 1:
                msg = f"{msg} (in {count} of {n_units} leave-one-out purifications)"
            warnings.warn(msg, w.category, stacklevel=4)

    @classmethod
    def _anchor_tc_draws(cls, ref, foc, Va, Vo, n_a, n_o, selection_method, corr_tol, sd_ratio_tol, min_anchors,
                         wald_alpha):
        """
        Translation constant for each draw (row) of the candidate anchors' reference values ref and focal values
        foc (draws x candidates), with the anchor selection redone in every draw: the same rules as
        _wald_anchor_selection (sequential exclusion on the t statistics of _anchor_wald_stats, from the
        observed covariance matrices Va (reference) and Vo (focal) and person counts n_a, n_o) and
        _robust_anchor_selection (modified z-scores of the deviations, dropped until the correlation and SD ratio
        clear their tolerances), and the same min_anchors floor. Returns an array of length draws.
        """
        B, n = ref.shape
        keep = np.ones((B, n), bool)
        if selection_method == "wald":
            d = ref - foc
            if n > min_anchors:
                for _ in range(n - min_anchors):
                    _, _, _, t, _, p = cls._anchor_wald_stats(d, Va, Vo, keep, n_a, n_o)
                    bad = ((p <= wald_alpha) & keep).any(1) & (keep.sum(1) > min_anchors)
                    if not bad.any():
                        break
                    worst = np.where(keep, np.abs(t), -1.0).argmax(1)
                    keep[np.where(bad)[0], worst[bad]] = False
            return cls._anchor_wald_stats(d, Va, Vo, keep, n_a, n_o)[0]
        # robust_z
        if n > min_anchors:
            dev = foc - ref
            med = np.median(dev, axis=1, keepdims=True)
            mad = np.median(np.abs(dev - med), axis=1, keepdims=True)
            with np.errstate(divide="ignore", invalid="ignore"):
                rz = np.abs(0.6745 * (dev - med) / mad)
            rz = np.where(np.isfinite(rz), rz, 0.0)
            for b in range(B):
                k = keep[b]
                while True:
                    a_, o_ = ref[b, k], foc[b, k]
                    sa = a_.std(ddof=1)
                    corr = np.corrcoef(a_, o_)[0, 1] if sa > 0 and o_.std(ddof=1) > 0 else np.nan
                    ratio = o_.std(ddof=1) / sa if sa > 0 else np.nan
                    # same test as _robust_anchor_selection's loop (a NaN comparison counts as satisfied)
                    if not ((corr < corr_tol) or (ratio < 1 / sd_ratio_tol) or (ratio > sd_ratio_tol)):
                        break
                    if k.sum() <= min_anchors:
                        break
                    k[np.argmax(np.where(k, rz[b], -1.0))] = False
        n_k = keep.sum(1)
        return (np.where(keep, ref, 0.0).sum(1) - np.where(keep, foc, 0.0).sum(1)) / n_k

    def _purification_draws(self, cols, ref_col, focal_col, ref_se_col, focal_se_col, ref_boot, focal_boot,
                            ref_cov, focal_cov):
        """
        Paired draws of the reference and focal values of cols (each draws x len(cols)) for re-selecting the
        anchors, each group's from the same source as its standard errors: draws from a normal distribution
        centred on the estimates with the group's covariance matrix when it has one (analytic SEs), else its
        bootstrap resamples (rows with a non-finite value dropped), else a normal distribution with the squared
        SEs on the diagonal. Normal draws (fixed seed) number _RESELECT_DRAWS, or the other group's resamples
        when it uses the bootstrap; resamples are paired in order, the first min(B_ref, B_focal). None if a group
        has no finite source.
        """
        groups = ((ref_col, ref_se_col, ref_boot, ref_cov), (focal_col, focal_se_col, focal_boot, focal_cov))
        src = []
        for col, se_col, boot, cov in groups:
            if cov is not None and set(cols).issubset(cov.index):
                src.append(("normal", cov.loc[cols, cols].to_numpy(float)))
            elif boot is not None and set(cols).issubset(boot.columns):
                X = boot[cols].to_numpy(float)
                src.append(("boot", X[np.isfinite(X).all(1)]))
            else:
                src.append(("normal", np.diag(se_col.loc[cols].to_numpy(float) ** 2)))
        n_boot = [len(X) for kind, X in src if kind == "boot"]
        m = min(n_boot) if n_boot else self._RESELECT_DRAWS
        if m <= 2:
            return None
        rng = np.random.default_rng(0)
        out = []
        for (col, _, _, _), (kind, X) in zip(groups, src):
            if kind == "boot":
                out.append(X[:m])
                continue
            mu = col.loc[cols].to_numpy(float)
            if not (np.isfinite(mu).all() and np.isfinite(X).all()):
                return None
            out.append(rng.multivariate_normal(mu, X, size=m, method="eigh"))
        return tuple(out)

    def _purification_loo_tc(
        self,
        ref_col,
        focal_col,
        ref_se_col,
        focal_se_col,
        selection_method,
        corr_tol,
        sd_ratio_tol,
        min_anchors,
        wald_alpha,
        ref_boot=None,
        focal_boot=None,
        ref_cov=None,
        focal_cov=None,
        n_ref=None,
        n_focal=None,
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
        - otherwise from the independence formula sum(a_i^2 se_i^2).

        With selection_method='wald' the anchors are selected on both
        groups' errors: each group's covariance matrix (ref_cov/focal_cov,
        else the covariance of its bootstrap resamples, else its squared
        SEs on the diagonal) and person count (n_ref/n_focal, for the
        Welch-Satterthwaite df of the t reference); see
        _wald_anchor_selection.

        Returns
        -------
        tc_per_unit : pandas.Series, indexed like ref_col -- computed
            WITHOUT r's (ref_col[r], focal_col[r]) pair.
        ref_var_per_unit, focal_var_per_unit : pandas.Series --
            Var(ref_col[r] - anchor_mean(ref, sel)) and
            Var(focal_col[r] - anchor_mean(focal, sel)) respectively;
            se = sqrt(ref_var_per_unit + focal_var_per_unit).
        selected, anchor_table : the whole-set (not leave-one-out) anchor
            selection, for reporting only; not used for the values above.
        """
        idx = ref_col.index
        # each group's covariance matrix of the units for the Wald selection: analytic, else bootstrap, else diagonal
        sel_cov = {}
        if selection_method == "wald":
            for key, cov, boot, se_col in (("ref", ref_cov, ref_boot, ref_se_col),
                                           ("focal", focal_cov, focal_boot, focal_se_col)):
                if cov is not None and set(idx).issubset(cov.index):
                    sel_cov[key] = cov.loc[idx, idx]
                else:
                    sel_cov[key] = self._boot_cov(boot, idx)
                    if sel_cov[key] is None:
                        sel_cov[key] = pd.DataFrame(np.diag(se_col.loc[idx].to_numpy(float) ** 2),
                                                    index=idx, columns=idx)
        tc_per_unit = pd.Series(index=idx, dtype=float)
        ref_var_per_unit = pd.Series(index=idx, dtype=float)
        focal_var_per_unit = pd.Series(index=idx, dtype=float)
        # The anchor selection is run once per tested unit; its warnings are
        # collected and reported once each (see _emit_loo_warnings).
        with warnings.catch_warnings(record=True) as loo_caught:
            warnings.simplefilter("always")
            for r in idx:
                others = idx.drop(r)
                ref_o, focal_o = ref_col.loc[others], focal_col.loc[others]
                ref_se_o, focal_se_o = ref_se_col.loc[others], focal_se_col.loc[others]
                w = None
                if selection_method == "wald":
                    ar = self._wald_anchor_selection(
                        ref_o,
                        focal_o,
                        focal_se_o,
                        alpha=wald_alpha,
                        min_anchors=min_anchors,
                        anchor_cov=sel_cov["ref"],
                        observed_cov=sel_cov["focal"],
                        n_anchor=n_ref,
                        n_observed=n_focal,
                    )
                    sel = ar["selected_anchors"]
                    tc_per_unit[r] = ar["tc"]
                    w = ar["weights"].loc[sel]
                elif selection_method == "robust_z":
                    ar = self._robust_anchor_selection(
                        ref_o,
                        focal_o,
                        corr_tol=corr_tol,
                        sd_ratio_tol=sd_ratio_tol,
                        min_anchors=min_anchors,
                    )
                    sel = ar["selected_anchors"]
                    tc_per_unit[r] = ar["tc"]
                else:
                    sel = others
                    tc_per_unit[r] = ref_o.mean() - focal_o.mean()
                n_sel = len(sel)

                have_boot = (
                    ref_boot is not None
                    and focal_boot is not None
                    and n_sel > 0
                    and set(sel).issubset(ref_boot.columns)
                    and set(sel).issubset(focal_boot.columns)
                    and r in ref_boot.columns
                    and r in focal_boot.columns
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
                    ref_cov is not None
                    and focal_cov is not None
                    and n_sel > 0
                    and set(sel).issubset(ref_cov.index)
                    and set(sel).issubset(focal_cov.index)
                    and r in ref_cov.index
                    and r in focal_cov.index
                ):
                    a = (
                        (w / w.sum()).reindex(sel).values
                        if w is not None
                        else np.full(n_sel, 1.0 / n_sel)
                    )
                    for cov, store in (
                        (ref_cov, ref_var_per_unit),
                        (focal_cov, focal_var_per_unit),
                    ):
                        Vss = cov.loc[sel, sel].values
                        Vsr = cov.loc[sel, r].values
                        store[r] = float(cov.loc[r, r] - 2.0 * a @ Vsr + a @ Vss @ a)
                elif n_sel > 0:
                    a = (w / w.sum()).reindex(sel).values if w is not None else np.full(n_sel, 1.0 / n_sel)
                    tc_var = float(
                        (a**2 * (ref_se_o.loc[sel] ** 2 + focal_se_o.loc[sel] ** 2).values).sum()
                    )
                    ref_var_per_unit[r] = ref_se_col.loc[r] ** 2
                    focal_var_per_unit[r] = focal_se_col.loc[r] ** 2 + tc_var
                else:
                    ref_var_per_unit[r] = np.nan
                    focal_var_per_unit[r] = np.nan
                if (
                    self._RESELECT_ANCHORS
                    and selection_method in ("wald", "robust_z")
                    and len(others) > min_anchors  # otherwise nothing can be dropped: the fixed-set variance is exact
                    and n_sel > 0
                    and np.isfinite(ref_var_per_unit[r] + focal_var_per_unit[r])
                ):
                    cols = list(others) + [r]
                    draws = self._purification_draws(
                        cols, ref_col, focal_col, ref_se_col, focal_se_col,
                        ref_boot, focal_boot, ref_cov, focal_cov,
                    )
                    if draws is not None:
                        R_d, F_d = draws
                        # null for r and the kept anchors: expected focal value = reference - tc
                        null_units = [u for u in cols if u in set(sel) or u == r]
                        shift = np.array([
                            (ref_col.loc[u] - tc_per_unit[r] - focal_col.loc[u]) if u in null_units else 0.0
                            for u in cols
                        ])
                        F_d = F_d + shift
                        if selection_method == "wald":
                            Va_o = sel_cov["ref"].loc[others, others].to_numpy(float)
                            Vo_o = sel_cov["focal"].loc[others, others].to_numpy(float)
                        else:
                            Va_o = Vo_o = None
                        tc_d = self._anchor_tc_draws(
                            R_d[:, :-1], F_d[:, :-1], Va_o, Vo_o, n_ref, n_focal, selection_method,
                            corr_tol, sd_ratio_tol, min_anchors, wald_alpha,
                        )
                        v = float(np.var(F_d[:, -1] + tc_d - R_d[:, -1], ddof=1))
                        # keep the split between the groups (Welch df, size_adjust) and rescale to the total
                        scale = v / (ref_var_per_unit[r] + focal_var_per_unit[r])
                        ref_var_per_unit[r] *= scale
                        focal_var_per_unit[r] *= scale

        with warnings.catch_warnings(record=True) as full_caught:
            warnings.simplefilter("always")
            if selection_method == "wald":
                ar_full = self._wald_anchor_selection(
                    ref_col,
                    focal_col,
                    focal_se_col,
                    alpha=wald_alpha,
                    min_anchors=min_anchors,
                    anchor_cov=sel_cov["ref"],
                    observed_cov=sel_cov["focal"],
                    n_anchor=n_ref,
                    n_observed=n_focal,
                )
                selected, anchor_table = ar_full["selected_anchors"], ar_full["table"]
            elif selection_method == "robust_z":
                ar_full = self._robust_anchor_selection(
                    ref_col,
                    focal_col,
                    corr_tol=corr_tol,
                    sd_ratio_tol=sd_ratio_tol,
                    min_anchors=min_anchors,
                )
                selected, anchor_table = ar_full["selected_anchors"], ar_full["table"]
            else:
                selected, anchor_table = idx, None

        self._emit_loo_warnings(loo_caught, full_caught, len(idx))

        return tc_per_unit, ref_var_per_unit, focal_var_per_unit, selected, anchor_table

    def _calibrate_anchor_core(
        self,
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
        std_errors_kwargs,
    ):
        """
        Shared core of calibrate_anchor() across SLM/PCM/RSM.

        The subclasses pass their calibrate()/std_errors() keyword
        arguments; each keeps its own threshold-compatibility test. MFRM
        has its own calibrate_anchor().

        Computes the translation constant (via adj / wald / robust_z /
        none selection), applies it to get anchor_items, resolves
        overwrite_anchors, builds anchor_summary, and (if applicable)
        triggers plot_anchor_selection() -- exactly the behaviour each
        subclass's calibrate_anchor() docstring documents under
        the same parameter names, this is a pure implementation-sharing
        extraction with no behaviour change.

        Callers must already have anchors as a pd.Series and self.items
        already calibrated (calibrate()'s re-run trigger differs
        slightly per subclass -- constant/threshold_constant/etc.
        -- so that step stays in each subclass's own calibrate_anchor(),
        before calling this).

        Parameters
        ----------
        std_errors_kwargs : dict
            Forwarded to self.std_errors(no_of_samples=no_of_samples,
            seed=seed, **std_errors_kwargs) if selection_method='wald'
            needs to auto-trigger it (self.item_se not already present).
            Carries whichever calibration kwarg name this subclass's
            calibrate()/std_errors() uses (e.g. {'constant':..., 'method':...,
            'matrix_power':..., 'log_lik_tol':...} for SLM/PCM, or the RSM
            equivalent with 'threshold_constant') -- exactly what each subclass's
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
            # the anchor values are fixed; the calibration's covariance and persons give the t statistics' SE and df
            result = self._wald_anchor_selection(
                anchors,
                self.items,
                self.item_se,
                alpha=wald_alpha,
                min_anchors=min_anchors,
                observed_cov=self._location_covariance(),
                n_observed=self.no_of_persons,
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
        self._anchor_kept = keep_given

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

    _SELECTION_DEPRECATED = ("likelihood", "warm_corr", "tolerance", "max_iters", "ext_score_adjustment")

    @staticmethod
    def _resolve_aic_sig_test(aic_relative_likelihood, aic_sig_test):
        """aic_sig_test= is the v1.4.0 name of aic_relative_likelihood= (FutureWarning; its value is used)."""
        if aic_sig_test is None:
            return aic_relative_likelihood
        warnings.warn(
            "aic_sig_test is deprecated and will be removed; use aic_relative_likelihood (same "
            "rule: the simpler model's relative likelihood compared with alpha, a likelihood ratio test at a level "
            "below alpha).",
            FutureWarning,
            stacklevel=3,
        )
        return bool(aic_sig_test)

    @staticmethod
    def _warn_selection_deprecated(passed):
        """model_selection()'s person-estimation and likelihood= arguments no longer apply (FutureWarning, ignored)."""
        if passed:
            warnings.warn(
                f"model_selection(): {', '.join(sorted(passed))} no longer appl"
                f"{'ies' if len(passed) == 1 else 'y'} and will be removed; the comparison now follows the "
                "estimation the object was calibrated with (see the docstring) and needs no person estimates.",
                FutureWarning,
                stacklevel=4,
            )

    @staticmethod
    def _stacked(rsm=None, pcm=None, names=None, K=None):
        """Stacked uncentred thresholds (item by item): RSM's delta_i + tau_h, or PCM's thresholds_uncentred."""
        if pcm is not None:
            return np.concatenate(
                [pcm.thresholds_uncentred.loc[n].dropna().to_numpy(float) for n in names]
            )
        d = rsm.items.reindex(names).to_numpy(float)
        tau = np.asarray(rsm.thresholds, float)[-K:]
        return (d[:, None] + tau[None, :]).ravel()

    # sampling='power''s default delta, in shares of the median category width: the same as the close-fit tolerance
    # default of MFRM.per_rater_model_selection (delta=0.1)
    _POWER_DELTA_DEFAULT = 0.1

    def _rsm_pcm_model_selection(self, test, aic_relative_likelihood, alpha, min_effect, no_of_samples, seed, delta,
                                 delta_units, sampling, power_threshold, n_subsamples, ridge=None,
                                 delta_measure="either"):
        """model_selection()'s RSM-vs-PCM comparison with its arguments checked: the comparison on the full data
        (_run_rsm_pcm_selection), then, with sampling=n, the likelihood ratio test refitted on random subsamples."""
        if isinstance(sampling, str) and sampling == "auto":
            sampling = None  # v1.4.0's default: subsampled only under the joint likelihood, which is no longer used
        dynamic = isinstance(sampling, str) and sampling == "dynamic"
        if dynamic:
            # v1.4.0's subsample size, now with both models refitted on each subsample
            from raschpy.rsm import RSM
            m_max = (int(self.max_score) if isinstance(self, RSM)
                     else int(np.nanmax(np.asarray(self.max_score_vector, float))))
            sampling = int(min(20 * (len(self.item_names) - 1) * max(m_max - 1, 1), 1500))
        elif isinstance(sampling, str):
            if sampling != "power":
                raise ValueError("sampling must be None, 'power', 'dynamic' or a whole number of persons.")
        elif sampling is not None:
            if isinstance(sampling, bool) or int(sampling) != sampling or sampling < 2:
                raise ValueError("sampling must be None, 'power', 'dynamic' or a whole number of persons (at least 2).")
            sampling = int(sampling)
        if sampling is not None and test not in (None, "LR"):
            raise ValueError("model_selection(sampling=...) applies to test='LR' only (the default test=None "
                             "resolves to 'LR' when sampling is given).")
        if test is None and sampling is not None:
            test = "LR"
        delta_default = delta is None
        if delta is None:
            # the close-fit tolerance with every test (0.1 median category widths); sampling=n refits the plain test
            delta = 0.0 if isinstance(sampling, int) else self._POWER_DELTA_DEFAULT
        if delta < 0 or min_effect < 0:
            raise ValueError("delta and min_effect must be at least 0.")
        if delta_measure not in ("overall", "largest", "either"):
            raise ValueError("delta_measure must be 'overall', 'largest' or 'either'.")
        if delta_units not in ("logits", "width"):
            raise ValueError("delta_units must be 'logits' or 'width'.")
        if sampling == "power" and not delta > 0:
            raise ValueError("model_selection(sampling='power') needs delta > 0: the departure the study should "
                             "detect.")
        if isinstance(sampling, int) and delta > 0:
            raise ValueError("model_selection(): delta (the close-fit test) cannot be combined with sampling=n; use "
                             "one or the other.")
        if not 0 < power_threshold < 1:
            raise ValueError("power_threshold must be between 0 and 1.")
        if int(n_subsamples) != n_subsamples or n_subsamples < 1:
            raise ValueError("n_subsamples must be a whole number of at least 1.")
        if ridge is not None and getattr(self, "estimation", None) not in ("cml", "cp-full-rcl", "cpat-rcl"):
            raise ValueError(
                "model_selection(ridge=...) needs estimates from estimation='cml', 'cp-full-rcl' or 'cpat-rcl'; "
                f"these are from estimation={getattr(self, 'estimation', None)!r}."
            )
        self._run_rsm_pcm_selection(test, aic_relative_likelihood, alpha, min_effect, no_of_samples, seed,
                                    delta=float(delta), delta_units=delta_units, sampling=sampling,
                                    power_threshold=power_threshold, pcm_ridge=ridge, delta_measure=delta_measure,
                                    delta_default=delta_default)
        self.model_comparison_rsm_pcm_sampling = None
        if isinstance(sampling, int):
            self._subsample_rsm_pcm(sampling, int(n_subsamples), alpha, min_effect, no_of_samples, seed,
                                    pcm_ridge=ridge)
            self.model_comparison_rsm_pcm_sampling["dynamic"] = dynamic

    def _subsample_rsm_pcm(self, n, n_subsamples, alpha, min_effect, no_of_samples, seed, pcm_ridge=None):
        """sampling=n: the likelihood ratio test refitted on n persons drawn at random (both models, with the
        object's calibration settings), n_subsamples times; PCM is preferred if more than half the subsamples reject
        RSM (and min_effect holds on the full data). The seed is recorded so the draws can be repeated."""
        from raschpy.rsm import RSM

        names = list(self.item_names)
        X = self.responses.loc[:, names]
        N = len(X)
        rec = dict(n=n, n_subsamples=n_subsamples)
        s = self.model_comparison_rsm_pcm_lr_summary
        if n >= N:
            rec.update(decision="full data (n is at least the number of persons)")
            self.model_comparison_rsm_pcm_sampling = rec
            self.model_comparison_rsm_pcm_lr_summary = pd.concat(
                [s.drop("Preferred"), pd.Series({"Subsample size": n, "Subsamples": 0}), s[["Preferred"]]]
            ).rename(s.name)
            return
        if seed is None:
            seed = int(np.random.default_rng().integers(2**31 - 1))
        rng = np.random.default_rng(seed)
        kw = self._calibration_kw()
        is_rsm = isinstance(self, RSM)
        ps = []
        for _ in range(n_subsamples):
            Z = X.iloc[np.sort(rng.choice(N, n, replace=False))]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # the full data have already been reported on
                m = (type(self)(Z, max_score=int(self.max_score)) if is_rsm
                     else type(self)(Z, max_score_vector=self.max_score_vector.reindex(names)))
                m.calibrate(**kw)
                m._run_rsm_pcm_selection("LR", True, alpha, 0, no_of_samples, int(rng.integers(2**31 - 1)),
                                         delta_units="logits", pcm_ridge=pcm_ridge)
            ps.append(float(m.model_comparison_rsm_pcm_p))
        share = float(np.mean(np.asarray(ps) <= alpha))
        effect_ok = min_effect <= 0 or self.model_comparison_rsm_pcm_delta_hat >= min_effect
        preferred = "PCM" if (share > 0.5 and effect_ok) else "RSM"
        rec.update(decision="subsamples", seed=seed, p=ps, share_rejecting=share)
        self.model_comparison_rsm_pcm_sampling = rec
        self.model_comparison_rsm_pcm_lr_preferred = preferred
        extra = {"Subsample size": n, "Subsamples": n_subsamples, "Share rejecting RSM": share, "Seed": seed}
        if n_subsamples == 1:
            extra["p-value (subsample)"] = ps[0]
        self.model_comparison_rsm_pcm_lr_summary = pd.concat(
            [s.drop("Preferred"), pd.Series(extra), pd.Series({"Preferred": preferred})]
        ).rename(s.name)

    def _run_rsm_pcm_selection(self, test, aic_relative_likelihood, alpha, min_effect, no_of_samples, seed,
                               delta=0.0, delta_units="width", sampling=None, power_threshold=0.8, pcm_ridge=None,
                               delta_measure="either", delta_default=False):
        """
        model_selection()'s RSM-vs-PCM comparison on the object's estimation (see RSM.model_selection):
        'cml' -> LR of the conditional log-likelihoods, the other model fitted by CML; 'cp-full-rcl' or 'cpat-rcl'
        -> composite LR (on the full pattern set, or CPAT's patterns), the other model fitted with the same
        estimation; PAIR with method='log-lik' (PCM 'pair', RSM 'pair-cpat') -> composite LR within PAIR's
        Bradley-Terry objective (PCM's PAIR fit, and RSM as that objective's maximum under the restriction
        b = delta_i + tau_h); PAIR's other methods -> conditional parametric bootstrap. Sets model_comparison_rsm_pcm_*.
        """
        from raschpy import _selection, _progress
        from raschpy.pcm import PCM
        from raschpy.rsm import RSM

        is_rsm = isinstance(self, RSM)
        est = getattr(self, "estimation", None)
        method = self._calibration_kw().get("method") or "log-lik"
        names = list(self.item_names)
        X = self.responses.loc[:, names]
        I = len(names)
        K = int(self.max_score) if is_rsm else int(self.max_score_vector.iloc[0])
        Kv = [K] * I
        scores = X.sum(axis=1)
        n_obs = X.notna().sum(axis=1)
        n_persons = int(((scores > 0) & (scores < n_obs * K)).sum())

        def other(cls, **kw):
            m = cls(X, max_score=K) if cls is RSM else cls(X, max_score_vector=Kv)
            m.calibrate(**kw)
            return m

        # pcm_ridge (model_selection(ridge=)): PCM fitted with that ridge for the comparison (a PCM object refitted
        # with its own other settings; its estimates are unchanged)
        if pcm_ridge is not None and not is_rsm:
            refit = PCM(X, max_score_vector=Kv)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # the object's own fit has already reported on the data
                refit.calibrate(**self._calibration_kw(ridge=pcm_ridge))
        ridge_kw = {} if pcm_ridge is None else dict(ridge=pcm_ridge)
        common = dict(test=test, aic_relative_likelihood=aic_relative_likelihood, alpha=alpha, min_effect=min_effect,
                      X=X, K=K, estimation=est, n_persons=n_persons, delta=delta, delta_units=delta_units,
                      sampling=sampling, power_threshold=power_threshold, delta_measure=delta_measure,
                      delta_default=delta_default)
        if est == "cml":
            rsm, pcm = ((self, other(PCM, estimation="cml", **ridge_kw)) if is_rsm
                        else (other(RSM, estimation="cml"), self if pcm_ridge is None else refit))
            if list(rsm.item_names) != names or list(pcm.item_names) != names:
                raise ValueError(
                    "model_selection() under CML needs CML estimates for every item in both RSM and PCM; items "
                    f"were removed ({sorted(set(names) - (set(rsm.item_names) & set(pcm.item_names)))}). "
                    "Use estimation='cp-full-rcl', which keeps them."
                )
            self._compare_rsm_pcm(self._stacked(rsm=rsm, names=names, K=K), self._stacked(pcm=pcm, names=names),
                                  "cml", detail=dict(ll_rsm=rsm.convergence["log_likelihood"],
                                                     ll_pcm=pcm.convergence["log_likelihood"]), **common)
            return
        if est in ("cp-full-rcl", "cpat-rcl"):
            rsm, pcm = ((self, other(PCM, estimation=est, **ridge_kw)) if is_rsm
                        else (other(RSM, estimation=est), self if pcm_ridge is None else refit))
            self._compare_rsm_pcm(self._stacked(rsm=rsm, names=names, K=K), self._stacked(pcm=pcm, names=names),
                                  "composite", **common)
            return
        pair_kw = dict(estimation="pair", method=method)
        if est in ("pair", "pair-cpat") and method == "log-lik":
            pcm = self if not is_rsm else other(PCM, **pair_kw)
            mat = getattr(pcm, "_pair_bt_matrix", None)
            if mat is not None:
                A0, _ = _selection.bases(I, K)
                b_rsm = _selection.bt_restricted_fit(mat, A0)
                self._compare_rsm_pcm(b_rsm, self._stacked(pcm=pcm, names=names), "composite", **common)
                return
        # no likelihood for this estimator: conditional parametric bootstrap, both models fitted with the object's
        # own estimator
        own = self._calibration_kw()
        rsm_kw = own if is_rsm else dict(estimation="pair-cpat", method=method)
        pcm_kw = pair_kw if is_rsm else own

        def fit_on(Z, cls, kw):
            m = cls(Z, max_score=K) if cls is RSM else cls(Z, max_score_vector=Kv)
            m.calibrate(**kw)
            return m

        def stat(Z, rsm=None, pcm=None):
            b0 = self._stacked(rsm=fit_on(Z, RSM, rsm_kw) if rsm is None else rsm, names=names, K=K)
            b1 = self._stacked(pcm=fit_on(Z, PCM, pcm_kw) if pcm is None else pcm, names=names)
            return _selection.threshold_difference(b0, b1, I, K)

        rsm = self if is_rsm else other(RSM, **rsm_kw)
        pcm = other(PCM, **pcm_kw) if is_rsm else self
        T = stat(X, rsm, pcm)
        # the datasets are simulated from RSM's 'cp-full-rcl' fit (the most accurate estimate of the null
        # distribution); the statistic is still the object's estimator's
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # the object's fit has already reported on the data
            b_sim = self._stacked(rsm=other(RSM, estimation="cp-full-rcl"), names=names, K=K)
        _, p, used = _selection.bootstrap_p(
            X, K, b_sim, stat, int(no_of_samples), np.random.default_rng(seed),
            track=lambda it: _progress.track(it, "Model selection bootstrap", unit="resamples"), observed=T,
        )
        self._compare_rsm_pcm(self._stacked(rsm=rsm, names=names, K=K), self._stacked(pcm=pcm, names=names),
                              "bootstrap", detail=dict(statistic=T, p=p, resamples=used), **common)

    def _compare_rsm_pcm(self, b_rsm, b_pcm, route, test, aic_relative_likelihood, alpha, min_effect, X, K, estimation,
                         n_persons, detail=None, delta=0.0, delta_units="width", sampling=None, power_threshold=0.8,
                         delta_measure="either", delta_default=False):
        """
        Set model_comparison_rsm_pcm_* from the comparison of route ('composite' or 'bootstrap', with its
        statistic and p in detail, or 'cml' with the log-likelihoods) at the stacked uncentred thresholds b_rsm
        and b_pcm. Also warns once when categories are thin or unused (see _selection.thin_categories).
        """
        from raschpy import _selection

        I = X.shape[1]
        names = list(X.columns)
        tau_rsm = (b_rsm - b_rsm.reshape(I, K).mean(1).repeat(K)).reshape(I, K)
        tau_pcm = (b_pcm - b_pcm.reshape(I, K).mean(1).repeat(K)).reshape(I, K)
        effect = float(np.abs(tau_pcm - tau_rsm).max())
        self.model_comparison_rsm_pcm_effect = effect
        thin, unused = _selection.thin_categories(X, K)
        if thin or unused:
            what = (
                f"score categor{'y' if len(unused) == 1 else 'ies'} {', '.join(map(str, unused))} "
                f"{'is' if len(unused) == 1 else 'are'} not used"
                if unused
                else f"the categories around threshold{'s' if len(thin) > 1 else ''} {', '.join(map(str, thin))} "
                f"have few responses"
            )
            warnings.warn(
                f"RSM vs PCM: {what}, so the comparison is not reliable for these data and tends to favour "
                "PCM (in simulations of RSM data with a rarely used middle category it rejected RSM in about 10% "
                "of datasets at alpha = 0.05, and almost always when the category was unused). No version of "
                "the test held its level in these data, and collapsing the category changes the model; check how "
                "the categories are used before choosing between the models.",
                UserWarning,
                stacklevel=4,
            )
        k_rsm = (I - 1) + (K - 1)
        k_pcm = I * K - 1
        info = dict(route=route, estimation=estimation)
        if route == "bootstrap":
            info["null_fit"] = "cp-full-rcl"
        if route == "cml":
            ll_rsm, ll_pcm = detail["ll_rsm"], detail["ll_pcm"]
            stat = max(2.0 * (ll_pcm - ll_rsm), 0.0)
            df = (I - 1) * (K - 1)
            p = float(chi2_dist.sf(stat, df))
            pen_rsm, pen_pcm = k_rsm, k_pcm
            extra = {"LR statistic": stat, "df": df}
        elif route == "composite":
            r = _selection.composite_lr(X, K, b_rsm, b_pcm, cpat=estimation == "cpat-rcl")
            stat, p, ll_rsm, ll_pcm = r["statistic"], r["p"], r["cl_rsm"], r["cl_pcm"]
            pen_rsm, pen_pcm = r["penalty_rsm"], r["penalty_pcm"]
            df = r["df"]
            extra = {"Composite LR statistic": stat, "df": df, "Scale": r["scale"],
                     "df (Satterthwaite)": r["df_satterthwaite"]}
            info.update({k: r[k] for k in ("weight_min", "weight_max")})
        else:  # bootstrap
            stat, p = detail["statistic"], detail["p"]
            df = (I - 1) * (K - 1)
            extra = {"Statistic": stat, "Resamples": detail["resamples"]}
            ll_rsm = ll_pcm = pen_rsm = pen_pcm = None
        self.model_comparison_rsm_pcm_info = info
        lab = {"cml": "LR test", "composite": "composite LR test", "bootstrap": "bootstrap test"}[route]

        # the departure from RSM: PCM's thresholds minus RSM's (both centred) over the item x threshold combinations;
        # its size is their root mean square weighted by 1 / (sampling variance of PCM's estimate) (_departure_size).
        # The test's distribution at a given size is computed from the responses a departure of that size would give
        # on average (_selection.CmlCloseFitNull, CompositeCloseFitNull); with a composite likelihood, the statistic is
        # a weighted sum of chi-squares (Kent's weights lam, and the observed departure's shift per component b; see
        # _selection.close_fit_parts) and the computed shift is spread over its components in proportion to b^2
        b0c, b1c = np.asarray(b_rsm, float) - np.mean(b_rsm), np.asarray(b_pcm, float) - np.mean(b_pcm)
        exact = cov = None  # cov: sampling covariance of PCM's centred thresholds (for the per-item test)
        if route == "cml":
            lam, b_obs, var, H = self._cml_close_fit_parts(X, K, b0c, b1c, hessian=True)
            Bm = np.hstack(_selection.bases(I, K))
            cov = Bm @ _selection._pinv_sym(H) @ Bm.T
            q0 = (I - 1) + (K - 1)
            exact = _selection.CmlCloseFitNull(X, K, b0c, b1c, lambda d, v=var: self._departure_size(d, v),
                                               rsm_info=H[:q0, :q0])
        elif route == "composite":
            (lam, b_obs), var, cov = r["parts"], r["var_pcm"], r.get("cov_pcm")
            if len(lam):
                exact = _selection.CompositeCloseFitNull(X, K, b0c, b1c, lambda d, v=var: self._departure_size(d, v),
                                                         lam, b_obs, cpat=estimation == "cpat-rcl")
        else:
            lam = b_obs = var = None
        rms = self._departure_size(b1c - b0c, var)
        if lam is None and delta_default:
            delta = 0.0  # the default tolerance: the bootstrap test has no close-fit version
        need_units = delta > 0 or min_effect > 0 or sampling == "power"
        try:
            unit, units = self._delta_unit(b0c, I, K, delta_units, need_units)
        except ValueError as e:
            if not (delta_default and min_effect <= 0 and sampling != "power"):
                raise
            warnings.warn(
                f"model_selection(): the default tolerance delta={delta:g} is in median category widths, but "
                f"{str(e).split(': ', 1)[1].split(';')[0]}, so the comparison has no tolerance (the plain test). To "
                "use one, set delta in logits with delta_units='logits'.",
                UserWarning,
                stacklevel=4,
            )
            delta = 0.0
            unit, units = self._delta_unit(b0c, I, K, delta_units, False)
        delta_logits = delta * unit
        b_u = None
        if lam is None:
            delta_hat = np.nan
            if need_units:
                raise ValueError(
                    "model_selection(): delta, min_effect and sampling='power' need a likelihood or composite "
                    f"likelihood; with estimation={estimation!r} and this method the comparison is a bootstrap test. "
                    "Use estimation='cp-full-rcl' or 'cml'."
                )
        elif exact is not None:
            # bias-corrected size: the size at which the statistic's expected excess over df equals the observed one
            delta_hat = exact.delta_hat(stat) / unit
        elif rms > 0 and len(lam):
            b_u = b_obs / rms
            shift = float((lam * b_u**2).sum())
            # bias-corrected size: the statistic's excess over its null mean, per unit of departure along b_u
            delta_hat = float(np.sqrt(max(stat - lam.sum(), 0.0) / shift)) / unit if shift > 0 else 0.0
        else:
            delta_hat = 0.0
        # the pieces, for close-fit and power calculations at other values of delta (private)
        self._rsm_pcm_close_fit_parts = dict(stat=stat, lam=lam, b_obs=b_obs, var=var, gap=b1c - b0c, unit=unit,
                                             units=units, n_persons=n_persons, exact=exact, df=df)
        effect_ok = min_effect <= 0 or (np.isfinite(delta_hat) and delta_hat >= min_effect)
        self.model_comparison_rsm_pcm_delta_hat = delta_hat
        self.model_comparison_rsm_pcm_delta_units = units
        size_rows = {"Delta-hat": delta_hat, "Units": units}
        # the close-fit test (every test, with a tolerance): RSM is rejected only if the departure is significantly
        # larger than delta (in the observed departure's direction), overall and/or for any one item
        self.model_comparison_rsm_pcm_close_fit_p = None
        self.model_comparison_rsm_pcm_close_fit_p_item = None
        self.model_comparison_rsm_pcm_item_close_fit = None
        close_rows, p_close = {}, None
        if delta_logits > 0 and sampling != "power" and lam is not None:
            if exact is not None:
                p_close = exact.p(stat, delta_logits)
            elif b_u is not None:
                p_close = _selection.nc_weighted_chi2_sf(stat, lam, (delta_logits * b_u) ** 2)
            else:
                p_close = float(p)
            self.model_comparison_rsm_pcm_close_fit_p = p_close
            close_rows = {"delta": delta, "p-value (close fit)": p_close}
            if delta_measure != "overall" and cov is not None:
                p_item, table = self._item_close_fit(b1c - b0c, cov, I, K, delta_logits, unit, names)
                self.model_comparison_rsm_pcm_close_fit_p_item = p_item
                self.model_comparison_rsm_pcm_item_close_fit = table
                close_rows["p-value (close fit, per item)"] = p_item
                p_close = p_item if delta_measure == "largest" else min(p_close, p_item)
        close_ok = p_close is None or p_close <= alpha
        if test is None:
            # the default: AIC where there is a likelihood, the bootstrap test (the only test) otherwise
            test = "LR" if route == "bootstrap" else "AIC"
        self.model_comparison_rsm_pcm_test = test
        if test == "LR" or route == "bootstrap":
            if route == "bootstrap" and test != "LR":
                raise ValueError(
                    f"model_selection(test={test!r}) needs a likelihood; with estimation={estimation!r} and this "
                    "method only the bootstrap test (test='LR') exists."
                )
            rows = {**extra, "p-value": p, **close_rows}
            p_decide = p if p_close is None else p_close
            self.model_comparison_rsm_pcm_power = None
            preferred = "PCM" if (p_decide <= alpha and effect_ok) else "RSM"
            if sampling == "power":
                # the sample size at which the test would detect a departure of size delta, in the observed
                # departure's direction, with probability power_threshold; with fewer persons than the data, the
                # decision is the one such a study would most likely reach: PCM if, at the estimated departure, it
                # would reject RSM with probability at least 0.5
                if exact is not None:
                    n_req = exact.sample_size(delta_logits, alpha, power_threshold, n_persons)
                elif b_u is not None:
                    n_req = _selection.power_sample_size(lam, b_u, delta_logits, alpha, power_threshold, n_persons)
                else:
                    n_req = np.inf
                pw = dict(delta=delta, units=units, power_threshold=power_threshold, n=n_req, n_data=n_persons)
                if n_req < n_persons:
                    if not delta_hat > 0:
                        proj = alpha
                    elif exact is not None:
                        proj = exact.power(delta_hat * unit, alpha, n_req / n_persons)
                    elif b_u is not None:
                        proj = _selection.close_fit_power(lam, b_u, delta_hat * unit, alpha, n_req / n_persons)
                    else:
                        proj = alpha
                    preferred = "PCM" if (proj >= 0.5 and effect_ok) else "RSM"
                    pw.update(decision="projected", projected_rejection=proj)
                    rows.update({"delta": delta, "Power n": n_req, "Projected rejection probability": proj})
                else:
                    pw.update(decision="full data")
                    rows.update({"delta": delta, "Power n": n_req if np.isfinite(n_req) else np.nan})
                self.model_comparison_rsm_pcm_power = pw
            self.model_comparison_rsm_pcm_lr = stat
            self.model_comparison_rsm_pcm_df = df
            self.model_comparison_rsm_pcm_p = p
            self.model_comparison_rsm_pcm_lr_preferred = preferred
            self.model_comparison_rsm_pcm_lr_summary = pd.Series(
                {**rows, **size_rows, "Preferred": preferred}, name=f"RSM vs PCM {lab}"
            )
        elif test == "AIC":
            aic_pcm, aic_rsm = 2 * pen_pcm - 2 * ll_pcm, 2 * pen_rsm - 2 * ll_rsm
            self.model_comparison_rsm_pcm_aic = {"PCM": aic_pcm, "RSM": aic_rsm}
            row = {"PCM AIC": aic_pcm, "RSM AIC": aic_rsm}
            if aic_relative_likelihood:
                d_aic = aic_rsm - aic_pcm
                gate = None
                if route == "composite" and d_aic > 0:
                    # the composite LR test of RSM within PCM, at the tail probability AIC's rule has with a
                    # likelihood: p is the relative likelihood that a likelihood ratio with the same chi-square(df)
                    # tail probability would give (a composite likelihood has no fixed scale)
                    tail = r["p"]
                    d_aic = (float(chi2_dist.isf(tail, df)) if tail > 0 else np.inf) - 2 * df
                    gate = {"composite LR": stat, "df": df, "tail p": float(tail)}
                rel_lik = float(np.exp(-max(d_aic, 0.0) / 2))
                preferred = "PCM" if (d_aic > 0 and rel_lik <= alpha and close_ok and effect_ok) else "RSM"
                # the rule is a likelihood ratio test of RSM within PCM at this level (stricter than alpha)
                level = float(chi2_dist.sf(2 * df + 2 * np.log(1 / alpha), df))
                self.model_comparison_rsm_pcm_aic_relative_likelihood = rel_lik
                self.model_comparison_rsm_pcm_aic_p = rel_lik  # v1.4.0 name, kept as an alias
                self.model_comparison_rsm_pcm_aic_level = level
                self.model_comparison_rsm_pcm_aic_gate = gate
                row["Relative likelihood"] = rel_lik
                row["Test level"] = level
            else:
                preferred = "PCM" if (aic_pcm < aic_rsm and close_ok and effect_ok) else "RSM"
            self.model_comparison_rsm_pcm_aic_preferred = preferred
            self.model_comparison_rsm_pcm_aic_summary = pd.Series(
                {**row, **close_rows, **size_rows, "Preferred": preferred},
                name="RSM vs PCM AIC comparison" + (" (composite)" if route == "composite" else ""),
            )
        else:  # BIC
            lg = np.log(n_persons)
            bic_pcm = (pen_pcm if route == "composite" else k_pcm) * lg - 2 * ll_pcm
            bic_rsm = (pen_rsm if route == "composite" else k_rsm) * lg - 2 * ll_rsm
            preferred = "PCM" if (bic_pcm < bic_rsm and close_ok and effect_ok) else "RSM"
            self.model_comparison_rsm_pcm_bic = {"PCM": bic_pcm, "RSM": bic_rsm}
            self.model_comparison_rsm_pcm_bic_preferred = preferred
            self.model_comparison_rsm_pcm_bic_summary = pd.Series(
                {"PCM BIC": bic_pcm, "RSM BIC": bic_rsm, **close_rows, **size_rows, "Preferred": preferred},
                name="RSM vs PCM BIC comparison" + (" (composite)" if route == "composite" else ""),
            )

    # how the departure's size weights the item x threshold combinations: 'information' (weights 1 / sampling
    # variance of the larger model's estimates; the close-fit test holds its level only with these, 2026-11-16
    # simulations) or 'none' (unweighted root mean square, for comparison)
    _DELTA_WEIGHTS = "information"

    def _departure_size(self, r, var):
        """Size of a departure r (item x threshold cells): root mean square, unweighted or (_DELTA_WEIGHTS =
        'information') weighted by the inverse sampling variances var of the larger model's estimates."""
        r = np.asarray(r, float)
        if self._DELTA_WEIGHTS == "information" and var is not None:
            v = np.asarray(var, float)
            w = np.where(v > 0, 1.0 / np.where(v > 0, v, 1.0), 0.0)
            if w.sum() > 0:
                return float(np.sqrt((w * r**2).sum() / w.sum()))
        return float(np.sqrt(np.mean(r**2)))

    def _item_close_fit(self, gap, cov, I, K, delta_logits, unit, names):
        """The per-item close-fit test (delta_measure 'largest' or 'either'): for each item, its PCM thresholds minus
        RSM's, centred on the item's own mean, sized as in _departure_size (in units), and a Wald statistic with
        PCM's sampling covariance; under 'departure = delta' along the observed direction it is noncentral
        chi-square(K - 1) with noncentrality W (delta / size)^2. Returns the Bonferroni p over items (the smallest
        p times I, at most 1) and a table of each item's departure and p."""
        from scipy.stats import ncx2

        Pk = np.eye(K) - 1.0 / K
        size, pv = [], []
        for i in range(I):
            sl = slice(i * K, (i + 1) * K)
            c = Pk @ np.asarray(gap, float)[sl]
            S = Pk @ cov[sl, sl] @ Pk
            sz = self._departure_size(c, np.diag(S))
            W = float(c @ np.linalg.pinv(S, rcond=1e-10) @ c)
            pv.append(float(ncx2.sf(W, K - 1, W * (delta_logits / sz) ** 2)) if sz > 0 and K > 1 else 1.0)
            size.append(sz / unit)
        table = pd.DataFrame({"Departure": size, "p-value": pv}, index=pd.Index(names, name="Item"))
        return float(min(1.0, I * min(pv))), table

    @staticmethod
    def _cml_close_fit_parts(X, K, b_rsm, b_pcm, hessian=False):
        """close_fit_parts for the CML likelihood ratio test of RSM within PCM: PCM's conditional information at
        its estimates (J = H), and the gap between the two fits, in the coordinates of _selection.bases."""
        from raschpy import _esf, _selection

        Xa = np.asarray(X, float)
        I = Xa.shape[1]
        _, _, Hs = _esf.cond_loglik(Xa, [b_pcm[i * K:(i + 1) * K] for i in range(I)], hessian=True)
        A0, E = _selection.bases(I, K)
        B = np.hstack([A0, E])
        H = -(B.T @ Hs @ B)
        gap = np.linalg.lstsq(B, b_pcm - b_rsm, rcond=None)[0]
        var = np.einsum("ij,jk,ik->i", B, _selection._pinv_sym(H), B)  # PCM's (centred) threshold variances
        out = (*_selection.close_fit_parts(H, H, A0.shape[1], gap), var)
        return (*out, H) if hessian else out

    @staticmethod
    def _delta_unit(b_rsm, I, K, delta_units, needed):
        """(logits per unit, units) for delta, min_effect and Delta-hat: 1 for 'logits'; for 'width', the median of
        RSM's category widths (the gaps between consecutive shared thresholds). If the widths are unusable (maximum
        score 1, or disordered thresholds), an error when a unit is needed, else logits."""
        if delta_units not in ("logits", "width"):
            raise ValueError("delta_units must be 'logits' or 'width'.")
        if delta_units == "logits":
            return 1.0, "logits"
        tau = np.asarray(b_rsm, float).reshape(I, K)[0]
        widths = np.diff(tau - tau.mean())
        if K >= 2 and (widths > 0).all():
            return float(np.median(widths)), "width"
        if needed:
            why = ("the maximum score is 1, so there are no category widths" if K < 2 else
                   f"RSM's thresholds are disordered (category widths {np.round(widths, 2).tolist()})")
            raise ValueError(f"delta_units='width': {why}; use delta_units='logits'.")
        return 1.0, "logits"

    def _resolve_fix_anchors(self, fix_anchors):
        """calibrate_anchor(fix_anchors=): None means True for CML estimates, else False; True needs CML."""
        cml = getattr(self, "estimation", None) == "cml"
        if fix_anchors is None:
            return cml
        if fix_anchors and not cml:
            raise ValueError(
                f"fix_anchors=True needs CML estimates (calibrate(estimation='cml'), or "
                f"calibrate_anchor(calibrate=True) after it); these were estimated with "
                f"estimation={getattr(self, 'estimation', None)!r}, for which anchoring is a "
                f"translation (fix_anchors=False)."
            )
        return bool(fix_anchors)

    def _fit_fixed_anchor_cml(self, anchors):
        """
        calibrate_anchor(fix_anchors=True): CML with the anchor items that keep their supplied values
        (self._anchor_kept, from _calibrate_anchor_core) held at those values and every other parameter
        estimated given them, starting from the translated estimates. Sets anchor_items to the fitted item
        locations and anchor_cml to the fit (fixed items, item locations and their SEs from the Hessian,
        thresholds, log-likelihood, convergence). With no anchors kept (overwrite_anchors='all'), nothing
        is fixed: anchor_items stays the translation and anchor_cml is None.
        """
        from types import SimpleNamespace
        from raschpy import _esf

        fixed = pd.Index([n for n in self.item_names if n in set(self._anchor_kept)])
        self.anchor_cml = None
        if not len(fixed):
            return
        X, A, c, b0, Ks, decode = self._cml_anchor_design(anchors.loc[fixed])
        # PCM's CML ridge: the same penalty (unchanged by the fixed item locations, constants per item)
        Pb = getattr(self, "_cml_Pb", None)
        Q = None if Pb is None else A.T @ Pb @ A
        fit = _esf.fit_cml(
            X, A, b0, Ks, offset=c, solver=getattr(self, "_cml_solver", None), penalty=Q
        )
        off = np.concatenate([[0], np.cumsum(Ks)])
        H = fit["hessian"]
        if H is None:
            H = _esf.cond_loglik(
                X,
                [fit["b"][off[i] : off[i + 1]] for i in range(len(Ks))],
                hessian=True,
            )[2]
        cov_b = A @ np.linalg.solve(-(A.T @ H @ A) + (0 if Q is None else Q), A.T)
        items, thresholds, L = decode(fit["b"])
        cov_b = np.nan_to_num(cov_b)  # markers (unused_categories='null') are fixed: no variance
        se = np.sqrt(np.clip(np.einsum("ip,pq,iq->i", L, cov_b, L), 0, None))
        se[items.index.isin(fixed)] = 0.0
        self.anchor_items = items
        self.anchor_cml = SimpleNamespace(
            fixed_items=fixed,
            items=items,
            item_se=pd.Series(se, index=items.index),
            thresholds=thresholds,
            log_likelihood=fit["ll"],
            convergence=dict(
                iterations=fit["iterations"],
                converged=fit["converged"],
                solver=fit["solver"],
                hessians=fit["hessians"],
            ),
        )
        summary = self.anchor_summary.copy()
        summary["Anchors fixed (CML)"] = len(fixed)
        self.anchor_summary = summary
        if not fit["converged"]:
            warnings.warn(
                f"The CML fit with the anchors fixed did not converge in {fit['iterations']} iterations "
                f"(solver {fit['solver']!r}, largest gradient {fit['max_gradient']:.2g}); anchor_items may "
                f"be unreliable. calibrate_anchor(fix_anchors=False) gives the translation instead.",
                UserWarning,
                stacklevel=3,
            )

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
            anchor_cml=getattr(self, "anchor_cml", None),
        )
        if extra:
            fields.update(extra)
        self.calibrate_anchor_runs[key] = SimpleNamespace(**fields)

    @staticmethod
    def _dif_welch(welch):
        """
        dif_test()'s deprecated welch argument (SLM, PCM, RSM): None (the default) or True gives Welch's t-test,
        False the normal (z) reference until the argument is removed; any value given raises a FutureWarning.
        Returns the bool the Wald tests use.
        """
        if welch is None:
            return True
        warnings.warn(
            "dif_test(welch=...) is deprecated and will be removed in a future version. The Wald tests now refer "
            "their statistics to a t-distribution with Welch-Satterthwaite degrees of freedom, which is more "
            "accurate than the normal distribution (z-test): it allows for estimating the standard errors from "
            "the groups' persons, which the normal reference ignores. "
            + ("welch=False still gives the normal (z) reference until then." if welch is False else
               "welch=True is the default.")
            + " Drop the argument to use the t reference.",
            FutureWarning,
            stacklevel=3,
        )
        return bool(welch)

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
            Each estimate's standard error, same shape as diff.
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
            Two-sided p-value from the t-distribution at each row's df.
        """
        se1 = np.asarray(se1, dtype=float)
        se2 = np.asarray(se2, dtype=float)
        diff = np.asarray(diff, dtype=float)
        combined_var = se1**2 + se2**2
        t_stat = diff / np.sqrt(combined_var)
        df = combined_var**2 / (se1**4 / (n1 - 1) + se2**4 / (n2 - 1))
        p = 2 * t_dist.sf(np.abs(t_stat), df)
        return t_stat, df, p

    def _resolve_andersen_groups(
        self, split_by, covariate, non_extreme, person_location_values, score_values
    ):
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
            raise ValueError(
                "split_by must be 'person_location', 'score', or 'exogenous'"
            )

        if split_by == "exogenous":
            if covariate is None:
                raise ValueError(
                    "covariate must be specified when split_by='exogenous'"
                )
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

        split_var = (
            person_location_values if split_by == "person_location" else score_values
        ).loc[non_extreme]
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

    @staticmethod
    def _items_arg(items, item, method):
        """The items argument of icc()/iic(), accepting the deprecated name item= with a FutureWarning."""
        if item is not None:
            warnings.warn(
                f"{method}(item=...) is deprecated and will be removed in a future release; "
                f"use {method}(items=...) (one item name or a list).",
                FutureWarning,
                stacklevel=3,
            )
            if items is not None:
                raise TypeError(f"{method}() got both items= and the deprecated item=.")
            items = item
        if items is None:
            raise TypeError(f"{method}() missing required argument: 'items'")
        return items

    def _cml_design(self, n_thresholds):
        """The design A of the CML thresholds b = A @ theta (+ the unused-category markers): the fit's
        after unused_categories='null', else the model's basis (RSM's, or a zero-sum basis)."""
        from raschpy import _rcl

        if getattr(self, "_cml_marked", None) is not None:
            return self._cml_A
        if hasattr(self, "_cml_basis"):
            return self._cml_basis()[0]
        return _rcl.zero_sum_basis(n_thresholds)

    def _person_score_bounds(self, df):
        """Each person's lowest and highest possible total over the items they responded to in df (from
        _score_bounds: 0 and each item's maximum score, or after calibrate(unused_categories='null') its
        lowest and highest used category). Persons at either bound have extreme scores."""
        lo, hi = self._score_bounds(list(df.columns))
        obs = df.notna().astype(float)
        return obs.mul(lo, axis=1).sum(axis=1), obs.mul(hi, axis=1).sum(axis=1)

    def _score_bounds(self, items):
        """(lowest, highest) possible score of each item (pandas series): 0 and its maximum score, or
        after calibrate(unused_categories='null') its lowest and highest used category."""
        hi = self._cml_max_scores(list(items))
        lo = np.zeros(len(hi))
        used = (
            getattr(self, "_cml_used", None)
            if getattr(self, "_cml_marked", None) is not None
            else None
        )
        if used:
            for j, n in enumerate(items):
                u = used.get(n)
                if u is not None:
                    lo[j] = float(np.argmax(u))
                    hi[j] = float(len(u) - 1 - np.argmax(u[::-1]))
        return pd.Series(lo, index=list(items)), pd.Series(hi, index=list(items))

    @staticmethod
    def _check_unused_categories(unused_categories, estimation, allowed=("drop", "null")):
        """calibrate(unused_categories=...) (PCM, RSM): one of allowed ('drop', 'null', and for PCM
        'ridge'); all but 'drop' with CML only."""
        if unused_categories not in allowed:
            names = ", ".join(repr(a) for a in allowed)
            raise ValueError(
                f"unused_categories must be one of {names}, got {unused_categories!r}."
            )
        if unused_categories != "drop" and estimation != "cml":
            raise ValueError(
                f"unused_categories={unused_categories!r} applies to estimation='cml' only (the other "
                "estimators keep every item, their smoothing giving finite estimates)."
            )

    # set by a caller for the fit it is about to run (global_fit()'s subsets, bootstrap resamples)
    _KEEP_ON_REFIT = frozenset({"_pre_fit_attrs", "_cml_fixed_hessian", "_cml_unused_fit"})

    def _clear_fit_results(self):
        """
        Before a refit (SLM, PCM, RSM calibrate()): remove what was derived from the previous fit --
        standard errors, person estimates, fit statistics and their tables, anchored results, DIF,
        model selection and the like. Methods compute these when they are missing and reuse them when
        present, so without this a refit with other settings would keep reporting the previous fit's
        values. They are recomputed on demand from the new fit. What the object held before its first
        fit (the data and the constructor's settings) is kept. MFRM does the same per representation
        (MFRM._clear_model_results).
        """
        if "_pre_fit_attrs" not in vars(self):
            self._pre_fit_attrs = frozenset(vars(self))
            return
        for name in list(vars(self)):
            if name not in self._pre_fit_attrs and name not in self._KEEP_ON_REFIT:
                delattr(self, name)

    def _calibration_kw(self, **passed):
        """
        calibrate() arguments for refits of the object's data (bootstrap
        resamples, and the group models of dif_test() and global_fit()):
        those of its last calibrate() call, overridden by any passed
        explicitly (not None), so a refit uses the same method as the
        estimates it accompanies. constant=None stays None, re-resolving
        the data-adaptive default for each refit.
        """
        kw = dict(getattr(self, "_calibrate_args", {}))
        kw.update({k: v for k, v in passed.items() if v is not None})
        return kw

    @staticmethod
    def _eig_pca(R, labels):
        """
        Principal components of a correlation matrix R (labels: its rows): the eigendecomposition, as
        DataFrames (eigenvectors, eigenvalues, variance_explained, loadings) with len(labels) - 1
        components, eigenvalues in descending order (the first is the first contrast), each eigenvector's
        sign set so its largest-magnitude element is positive, variance explained = eigenvalue / trace, and
        loadings = eigenvector x sqrt(eigenvalue). Raises ValueError if R is not finite.
        """
        R = np.asarray(R, float)
        if not np.all(np.isfinite(R)):
            raise ValueError("The residual correlation matrix is not finite.")
        values, vectors = np.linalg.eigh(R)
        order = np.argsort(values)[::-1]
        values, vectors = values[order], vectors[:, order]
        signs = np.sign(
            vectors[np.abs(vectors).argmax(axis=0), np.arange(vectors.shape[1])]
        )
        vectors = vectors * np.where(signs == 0, 1.0, signs)
        k = len(labels)
        n = k - 1
        pc_labels = [f"PC {pc + 1}" for pc in range(n)]
        eigenvectors = pd.DataFrame(
            vectors[:, :n].T,
            index=pc_labels,
            columns=[f"Eigenvector {pc + 1}" for pc in range(k)],
        )
        eigenvalues = pd.DataFrame(values[:n], index=pc_labels, columns=["Eigenvalue"])
        variance_explained = pd.DataFrame(
            values[:n] / np.trace(R), index=pc_labels, columns=["Variance explained"]
        )
        loadings = pd.DataFrame(
            vectors[:, :n] * np.sqrt(np.clip(values[:n], 0, None)),
            index=labels,
            columns=pc_labels,
        )
        return eigenvectors, eigenvalues, variance_explained, loadings

    def _residual_pca(self):
        """res_corr_analysis(): principal components of the standardised residuals' correlations."""
        return self._eig_pca(self.std_residual_df.corr().values, self.responses.columns)

    @staticmethod
    def _wilson_hilferty_z(ms, q):
        """Standardised mean square: Wilson-Hilferty cube-root transformation with standard deviation q."""
        return (ms ** (1 / 3) - 1) * (3 / q) + q / 3

    @staticmethod
    def _restscore_rng(seed):
        """Random generator for the item-restscore Monte Carlo draws: its own stream, reproducible from an integer
        seed (distinct from the person p-values' stream); fresh entropy for None."""
        if isinstance(seed, (int, np.integer)):
            return np.random.default_rng([int(seed), 0x7E57])
        return np.random.default_rng(seed)

    def _restscore_table(self, g_obs, g_exp, g_se, sims, labels, p_method):
        """
        Item-restscore table (Kreiner): the observed gamma of each unit's score against its restscore, the gamma
        of the expected table (sum over persons of P(X = k | total) at restscore total - k; descriptive), the
        observed gamma's asymptotic SE (descriptive, as iarm's), and the conditional Monte Carlo test: sims holds
        the gammas of datasets drawn from the model given each person's total at the fitted thresholds (no
        refit); 'Simulated mean' and 'Simulated SD' are their mean and SD, z = (observed - simulated mean) /
        simulated SD, p two-sided ('studentised': normal p of z; 'monte-carlo': the share of draws at least as
        far from their mean), p (BH) Benjamini-Hochberg adjusted.
        """
        from raschpy import _esf

        mu, sd, z, p = _esf.restscore_p(np.asarray(g_obs, float), sims, p_method)
        tab = pd.DataFrame(
            {"Observed gamma": g_obs, "Expected gamma": g_exp, "SE": g_se, "Simulated mean": mu,
             "Simulated SD": sd, "z": z, "p": p},
            index=labels,
        )
        tab["p (BH)"] = self._bh_correction(tab["p"].fillna(1.0)).where(tab["p"].notna())
        return tab

    def _apply_conditional_fit(
        self, samples=1000, seed=None, bootstrap_samples=None, q1_groups=None,
        q1_overall_samples=99, q1_overall_exceedances=10, restscore_p_method="studentised",
    ):
        """
        fit_statistics(conditional=True): replace the expected scores, variances and fourth central moments
        with their conditional values given each person's total over the items they responded to (they do
        not involve the person locations), for the same non-extreme persons, and recompute the residuals and
        the item and person infit/outfit statistics from them (Christensen & Kreiner): the standardisation
        uses the exact conditional moments, and counts are of the responses with a conditional variance
        (a person with a single response has none). Outputs that need person locations (point-measure
        correlations, discrimination, CSEM/RSEM, separation and reliability) are left as computed.
        """
        from raschpy import _esf

        if restscore_p_method not in ("studentised", "monte-carlo"):
            raise ValueError(
                f"restscore_p_method must be 'studentised' or 'monte-carlo', got {restscore_p_method!r}."
            )
        names = list(self.item_names)
        X = np.array(self.responses.loc[:, names], dtype=np.float64)
        mom = _esf.conditional_moments(X, self._cml_threshold_list(names), tables=True)
        self._warn_conditional_unused(X, names)
        # the thresholds the simulations (restscore draws, bootstrap, overall Q1) are drawn at
        sim_thr, self.conditional_null_fit = self._conditional_null(names)
        idx = self.residual_df.index
        pos = self.responses.index.get_indexer(idx)
        frame = lambda a: pd.DataFrame(a[pos], index=idx, columns=names)  # noqa: E731
        E, V, M4 = frame(mom["E"]), frame(mom["V"]), frame(mom["M4"])
        # a response determined by the total has variance 0; rounding can leave ~1e-29
        V = V.where(V > COND_VAR_ZERO)
        resid = self.responses.loc[idx, names] - E
        resid = resid.where(V.notna())
        std = resid / V**0.5
        self.exp_score_df, self.info_df, self.kurtosis_df = E, V, M4.where(V.notna())
        self.residual_df, self.std_residual_df = resid, std
        n_item, n_person = std.notna().sum(axis=0), std.notna().sum(axis=1)
        ratio = self.kurtosis_df / V**2

        self.item_outfit_ms = (std**2).mean()
        self.item_infit_ms = (resid**2).sum() / V.sum()
        q = ((ratio.sum() / n_item**2) - 1 / n_item) ** 0.5
        self.item_outfit_zstd = self._wilson_hilferty_z(self.item_outfit_ms, q)
        q = ((self.kurtosis_df - V**2).sum() / V.sum() ** 2) ** 0.5
        self.item_infit_zstd = self._wilson_hilferty_z(self.item_infit_ms, q)

        self.person_outfit_ms = (std**2).mean(axis=1).rename("Outfit MS")
        self.person_infit_ms = ((resid**2).sum(axis=1) / V.sum(axis=1)).rename(
            "Infit MS"
        )
        q = ((ratio.sum(axis=1) / n_person**2) - 1 / n_person) ** 0.5
        self.person_outfit_zstd = self._wilson_hilferty_z(
            self.person_outfit_ms, q
        ).rename("Outfit Z")
        q = ((self.kurtosis_df - V**2).sum(axis=1) / V.sum(axis=1) ** 2) ** 0.5
        self.person_infit_zstd = self._wilson_hilferty_z(
            self.person_infit_ms, q
        ).rename("Infit Z")

        if hasattr(self, "item_residual_corr"):
            self.item_residual_corr = std.corrwith(self.items, axis=1)
        if hasattr(self, "person_residual_corr"):
            self.person_residual_corr = std.corrwith(self.persons.loc[idx], axis=0)
        # item-restscore association (Kreiner): Goodman-Kruskal gamma of each item's score against its
        # restscore, tested against its conditional Monte Carlo distribution (see _restscore_table)
        obs_all = ~np.isnan(X)
        Xi = np.where(obs_all, X, 0).astype(int)
        tot = Xi.sum(1)
        g_obs, g_se, g_exp = [], [], []
        for i, item in enumerate(names):
            who = obs_all[:, i]
            observed = np.zeros_like(mom["tables"][i])
            np.add.at(observed, (Xi[who, i], tot[who] - Xi[who, i]), 1.0)
            g, se = _esf.gk_gamma(observed)
            g_obs.append(g)
            g_se.append(se)
            g_exp.append(_esf.gk_gamma(mom["tables"][i])[0])
        Ks = [int(k) for k in self._cml_max_scores(names)]
        sims = _esf.restscore_monte_carlo(
            X, sim_thr, Ks, np.arange(len(names)), int(samples), self._restscore_rng(seed)
        )
        self.item_restscore_table = self._restscore_table(
            g_obs, g_exp, g_se, sims, names, restscore_p_method
        )
        # conditional item chi-square (Q1) on ability groups (or supplied groups)
        self._q1_labels = self._q1_group_labels(mom["V"], q1_groups)
        self._set_q1_tables(X, mom["E"], mom["V"], self._q1_labels, names)
        self._set_q1_overall(
            X, names, self._q1_labels, q1_overall_samples, q1_overall_exceedances, seed, sim_thr
        )
        # conditional person fit: probability of a pattern no more likely than the person's, given total
        pv = _esf.person_pattern_pvalues(
            X, self._cml_threshold_list(names), samples=samples, rng=seed
        )
        self.person_conditional_p = pd.Series(
            pv, index=self.responses.index, name="Conditional p"
        ).loc[idx]
        self._conditional_tables = mom["tables"]
        self.fit_conditional = True
        self.item_fit_bootstrap = self.q3_bootstrap = None
        self.residual_contrast_bootstrap = None
        if bootstrap_samples:
            self._conditional_fit_bootstrap(X, names, int(bootstrap_samples), seed, sim_thr)

    Q1_MIN_GROUP = 30  # smallest ability group for the conditional item chi-square (Q1)

    @staticmethod
    def _ability_groups(locations, groups=None, min_size=None):
        """
        Groups for the conditional item chi-square (Q1): integer labels 0..G-1 per person (-1 = left out)
        from person locations (NaN = left out), as G groups of about equal size ordered by location with
        tied locations kept together (so with complete data they are groups of adjacent total scores);
        groups smaller than min_size are merged into their neighbour while more than 2 groups remain, so
        small samples keep 2 groups (fewer only when every location is tied). groups=None chooses G from
        the number of persons (about one group per 2 * min_size persons, from 2 to 10). Returns the labels
        and each group's location range.
        """
        min_size = Rasch.Q1_MIN_GROUP if min_size is None else int(min_size)
        loc = np.asarray(locations, float)
        ok = np.isfinite(loc)
        n = int(ok.sum())
        G = int(np.clip(n // (2 * min_size), 2, 10)) if groups is None else int(groups)
        if G < 2:
            raise ValueError("The conditional item chi-square needs at least 2 groups.")
        labels = np.full(len(loc), -1)
        vals, inv, counts = np.unique(loc[ok], return_inverse=True, return_counts=True)
        before = np.concatenate([[0], np.cumsum(counts)[:-1]])
        g_of_val = np.minimum((G * (before + counts / 2.0) / n).astype(int), G - 1)
        # merge groups below min_size into the neighbour with fewer persons (ends: inward), keeping 2
        while True:
            sizes = np.bincount(g_of_val, weights=counts, minlength=g_of_val.max() + 1)
            present = np.flatnonzero(sizes > 0)
            small = [g for g in present if sizes[g] < min_size]
            if not small or len(present) <= 2:
                break
            g = small[0]
            k = int(np.searchsorted(present, g))
            nbrs = [present[j] for j in (k - 1, k + 1) if 0 <= j < len(present)]
            target = min(nbrs, key=lambda h: sizes[h])
            g_of_val[g_of_val == g] = target
        _, g_of_val = np.unique(g_of_val, return_inverse=True)
        labels[np.flatnonzero(ok)] = g_of_val[inv]
        ranges = [
            (float(vals[g_of_val == g].min()), float(vals[g_of_val == g].max()))
            for g in range(g_of_val.max() + 1)
        ]
        return labels, ranges

    @staticmethod
    def _q1_statistics(X, E, V, labels):
        """
        Conditional item chi-square (van den Wollenberg's Q1) per item: within each group (labels, -1 =
        left out) the observed item score total O against its conditional expectation E given each
        person's total and items, Q1 = sum over groups of (O - E)^2 / V (V the sum of the conditional
        variances), df = groups - 1. Only responses with a conditional variance count. Returns dict(q1,
        df, O, E, V, n) with q1/df (I,) and the rest (I, G).
        """
        G = int(labels.max()) + 1
        valid = np.isfinite(V) & (V > 0) & (labels[:, None] >= 0)
        I = X.shape[1]
        O_, E_, V_, n_ = (np.zeros((I, G)) for _ in range(4))
        for g in range(G):
            sel = valid & (labels[:, None] == g)
            O_[:, g] = np.where(sel, X, 0.0).sum(0)
            E_[:, g] = np.where(sel, E, 0.0).sum(0)
            V_[:, g] = np.where(sel, V, 0.0).sum(0)
            n_[:, g] = sel.sum(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            terms = np.where(V_ > 0, (O_ - E_) ** 2 / V_, 0.0)
        df = (V_ > 0).sum(1) - 1
        return dict(q1=terms.sum(1), df=df, O=O_, E=E_, V=V_, n=n_)

    def _q1_group_labels(self, V, q1_groups):
        """Person group labels for Q1 (persons in responses order; -1 = left out): ability groups from
        the person estimates by default (q1_groups=None or an int G), else the supplied labels (a Series
        indexed by person, or an array in responses order). Persons without any response with a
        conditional variance (extreme totals, single responses) are left out."""
        contributes = (np.isfinite(V) & (V > COND_VAR_ZERO)).any(axis=1)
        lo, hi = self._person_score_bounds(self.responses.loc[:, list(self.item_names)])
        tot = self.responses.loc[:, list(self.item_names)].sum(axis=1)
        contributes &= ((tot > lo) & (tot < hi)).to_numpy()  # extreme: no variance given the total
        if q1_groups is None or np.isscalar(q1_groups):
            loc = self.persons.reindex(self.responses.index).to_numpy(float)
            loc = np.where(contributes, loc, np.nan)
            labels, ranges = self._ability_groups(loc, groups=q1_groups)
            self._q1_ranges = ranges
            return labels
        lab = (
            q1_groups.reindex(self.responses.index)
            if isinstance(q1_groups, pd.Series)
            else pd.Series(list(q1_groups), index=self.responses.index)
        )
        codes, uniques = pd.factorize(lab)
        self._q1_ranges = None
        self._q1_group_names = list(uniques)
        return np.where(contributes, codes, -1)

    def local_dependence_gamma(
        self, alpha=0.05, p_method="permutation", samples=1000, seed=None
    ):
        """
        Partial gamma tests of local dependence for every ordered item pair (Kreiner & Christensen, 2004):
        Goodman and Kruskal's gamma between the scores on items i and j, counting concordant and
        discordant pairs of persons only within strata of i's rest score (the person's total over the other
        items they responded to, and which items those were), pooled over the strata (partial gamma).

        Under the Rasch model the other items carry information about a person's location only through
        their total, so given i's rest score item i is independent of every other item, whatever the
        distribution of person locations: the partial gamma is zero in expectation. The test needs no
        parameter estimates. A clearly positive gamma means i and j are more alike than the rest score
        explains (local dependence: response dependence, redundancy, a shared secondary dimension). Each
        pair is tested in both directions (conditioning on i's rest score, then on j's). A strongly
        dependent pair can also give one of its items negative gammas with other items (its rest score
        contains the dependent partner), so read positive flags first and recheck after resolving the
        strongest pair.

        Parameters
        ----------
        alpha : float, default 0.05
            Significance level for the Flagged column (on the BH-adjusted p-values).
        p_method : {'permutation', 'monte-carlo'}, default 'permutation'
            Within each stratum, item i's responses are exchangeable under the model, so the test
            refers gamma's numerator S = C - D (concordant minus discordant pairs) to its distribution
            under permutation of item i's responses within strata. 'permutation': z = S divided by the
            exact standard deviation of S under that permutation (Kendall's variance with ties, summed
            over strata), normal p-value; no simulation, so p-values are not limited by a number of
            samples. 'monte-carlo': p = (1 + #{|simulated gamma| >= |observed|}) / (1 + samples) from
            samples within-stratum permutations (its smallest p-value, 1 / (1 + samples), limits the
            BH-adjusted p-values when many pairs are tested). In null simulations (10 items; SLM,
            PCM, RSM; N = 300 and 1,000; complete, two-booklet and 20% missing completely at random;
            300 datasets per cell) the per-pair rejection rate at 0.05 was 0.046-0.052 for
            'permutation' (SD of z 0.99-1.01).
        samples : int, default 1000
            Permutations (p_method='monte-carlo').
        seed : int or None, default None
            Seed for the permutations (p_method='monte-carlo'); None draws fresh entropy.

        Returns
        -------
        pandas.DataFrame
            Also stored as local_dependence_gamma_table: one row per ordered pair -- Item (the item
            whose rest score is conditioned on), Paired item, Gamma, z (from p_method; for
            'monte-carlo', the 'permutation' z), p, p (BH) (Benjamini-Hochberg over all ordered pairs),
            Flagged -- sorted by Gamma, largest first. local_dependence_gamma_info records p_method,
            samples and seed.
        """
        from raschpy import _esf

        if p_method not in ("permutation", "monte-carlo"):
            raise ValueError("p_method must be 'permutation' or 'monte-carlo'.")
        rng = np.random.default_rng(seed)
        names = list(self.item_names)
        X = np.array(self.responses.loc[:, names], dtype=np.float64)
        obs = ~np.isnan(X)
        Xi = np.where(obs, X, 0).astype(int)
        Ks = [int(k) for k in self._cml_max_scores(names)]
        rows = []
        chunk = 100
        for i in range(len(names)):
            others = np.arange(len(names)) != i
            rest = Xi[:, others].sum(axis=1)
            _, pattern = np.unique(obs[:, others], axis=0, return_inverse=True)
            _, stratum = np.unique(
                np.column_stack([pattern.ravel(), rest]), axis=0, return_inverse=True
            )
            stratum = stratum.ravel()
            hit = np.flatnonzero(obs[:, i])
            perms = None
            if p_method == "monte-carlo":
                # within-stratum permutations of item i's responses (rows: samples, over hit persons)
                srt = np.argsort(stratum[hit], kind="stable")
                perms = np.empty((samples, len(hit)), int)
                for b in range(samples):
                    order = np.lexsort((rng.random(len(hit)), stratum[hit][srt]))
                    perms[b, srt] = srt[order]
            pos = np.full(len(X), -1)
            pos[hit] = np.arange(len(hit))
            for j in range(len(names)):
                if j == i:
                    continue
                who = np.flatnonzero(obs[:, i] & obs[:, j])
                if not len(who):
                    continue
                _, s_idx = np.unique(stratum[who], return_inverse=True)
                S = s_idx.max() + 1
                T = np.zeros((S, Ks[i] + 1, Ks[j] + 1))
                np.add.at(T, (s_idx, Xi[who, i], Xi[who, j]), 1.0)
                g = _esf.partial_gk_gamma(T)[0]
                C, D, var = _esf.stratified_kendall_s(T)
                z = (C - D) / np.sqrt(var) if var > 0 else np.nan
                if p_method != "monte-carlo" or not np.isfinite(g):
                    p = 2 * norm.sf(abs(z)) if np.isfinite(z) else np.nan
                else:
                    xi_hit = Xi[hit, i]
                    exceed = 0
                    for c0 in range(0, samples, chunk):
                        pb = perms[c0 : c0 + chunk][
                            :, pos[who]
                        ]  # permuted source positions
                        nb = pb.shape[0]
                        Tb = np.zeros((nb, S, Ks[i] + 1, Ks[j] + 1))
                        bb = np.repeat(np.arange(nb), len(who))
                        np.add.at(
                            Tb,
                            (
                                bb,
                                np.tile(s_idx, nb),
                                xi_hit[pb].ravel(),
                                np.tile(Xi[who, j], nb),
                            ),
                            1.0,
                        )
                        gb = np.nan_to_num(_esf.partial_gk_gamma_batch(Tb), nan=0.0)
                        exceed += int((np.abs(gb) >= abs(g) - 1e-12).sum())
                    p = (1 + exceed) / (1 + samples)
                rows.append((names[i], names[j], g, z, p))
        tab = pd.DataFrame(rows, columns=["Item", "Paired item", "Gamma", "z", "p"])
        tab["p (BH)"] = self._bh_correction(tab["p"].fillna(1.0)).where(
            tab["p"].notna()
        )
        tab["Flagged"] = tab["p (BH)"] <= alpha
        tab = tab.sort_values("Gamma", ascending=False, ignore_index=True)
        self.local_dependence_gamma_table = tab
        self.local_dependence_gamma_info = dict(
            p_method=p_method,
            samples=samples if p_method == "monte-carlo" else None,
            seed=seed,
        )
        return tab

    def targeting(self, grid=(-6.0, 6.0, 0.05)):
        """
        Targeting and population measurement properties of the test (all items), after Kreiner &
        Christensen (2013, ch. 4), from the item estimates and a normal distribution of person locations
        fitted to the observed total scores (maximising the marginal likelihood of each person's total
        over the items they responded to, item parameters fixed).

        The person estimates are weighted likelihood estimates (WLE) for each total on the full test,
        finite for the extreme totals; bias, SEM and RMSE at each location are their exact values over
        the distribution of the total there, not asymptotic approximations.

        Parameters
        ----------
        grid : tuple, default (-6.0, 6.0, 0.05)
            (from, to, step) of the locations for targeting_curves.

        Returns
        -------
        pandas.Series
            Also stored as targeting_summary:
            Population mean, Population SD -- the fitted normal distribution;
            Target -- the location of maximum test information (where the RMSE of the estimates is
            typically smallest); Target - population mean;
            Maximum information, Expected information (over the population), Expected / maximum
            information;
            Test difficulty -- the location with expected total half the maximum;
            Expected |bias|, Expected SEM, Expected RMSE -- of the WLE, over the population;
            rho1 = V(T) / V(R) (true-score over total-score variance); rho2 = corr(R1, R2) (two
            independent administrations; equal to rho1 when responses are independent given the
            location); rho3 = V(E(theta-hat | theta)) / V(theta-hat) (rho1 for the estimates);
            rho4 = corr(theta-hat1, theta-hat2) (equal to rho3 likewise); rho5 = corr(theta,
            theta-hat); rho6 = P(R1 < R2 | theta1 < theta2), the probability that two random
            persons' totals order them correctly.
            Also set: targeting_curves (by location: test information, asymptotic SE 1 / sqrt(I),
            bias, SEM and RMSE of the WLE, population density), targeting_score_table (by total:
            WLE, model and observed proportions -- observed among persons who responded to every
            item) and targeting_items (each item's target, the location of its maximum information,
            and that information).
        """
        from scipy.optimize import minimize_scalar
        from scipy.stats import norm as _norm
        from raschpy import _esf

        names = list(self.item_names)
        thr = self._cml_threshold_list(names)
        X = np.array(self.responses.loc[:, names], dtype=np.float64)
        obs = ~np.isnan(X)
        keep = obs.any(axis=1)
        pats, inv = _esf._patterns(obs[keep])
        tot = np.where(obs, X, 0).sum(axis=1)[keep].astype(int)
        lgs, counts = [], []
        for p_idx, pat in enumerate(pats):
            lg = _esf.log_gamma([thr[k] for k in np.flatnonzero(pat)])
            lgs.append(lg)
            counts.append(
                np.bincount(tot[inv == p_idx], minlength=len(lg)).astype(float)
            )
        mu, sigma = _esf.fit_normal_population(lgs, counts)
        lg_full = _esf.log_gamma(thr)
        pp = _esf.population_properties(lg_full, mu, sigma)
        self.targeting_summary = pd.Series(
            {
                "Population mean": mu,
                "Population SD": sigma,
                "Target": pp["target"],
                "Target - population mean": pp["target"] - mu,
                "Maximum information": pp["max_information"],
                "Expected information": pp["expected_information"],
                "Expected / maximum information": pp["expected_information_ratio"],
                "Test difficulty": pp["test_difficulty"],
                "Expected |bias|": pp["expected_abs_bias"],
                "Expected SEM": pp["expected_sem"],
                "Expected RMSE": pp["expected_rmse"],
                "rho1": pp["rho1"],
                "rho2": pp["rho2"],
                "rho3": pp["rho3"],
                "rho4": pp["rho4"],
                "rho5": pp["rho5"],
                "rho6": pp["rho6"],
            },
            name="Targeting",
        )
        est = pp["estimates"]
        est = np.where(np.isfinite(lg_full), est, np.nan)  # totals that cannot occur: no estimate
        th = np.arange(grid[0], grid[1] + grid[2] / 2, grid[2])
        P = _esf.score_distribution(lg_full, th)
        _, info, _ = _esf.score_moments(lg_full, th)
        est_c = np.nan_to_num(est)  # probability 0 where NaN
        mean_est = P @ est_c
        self.targeting_curves = pd.DataFrame(
            {
                "Information": info,
                "Asymptotic SE": 1 / np.sqrt(info),
                "Bias": mean_est - th,
                "SEM": np.sqrt(np.maximum(P @ est_c**2 - mean_est**2, 0)),
                "RMSE": np.sqrt((P * (est_c[None, :] - th[:, None]) ** 2).sum(1)),
                "Population density": _norm.pdf(th, mu, sigma),
            },
            index=pd.Index(np.round(th, 10), name="Location"),
        )
        full = obs.all(axis=1)
        observed = (
            np.bincount(
                np.where(obs, X, 0).sum(axis=1)[full].astype(int),
                minlength=len(lg_full),
            )
            / full.sum()
            if full.any()
            else np.full(len(lg_full), np.nan)
        )
        self.targeting_score_table = pd.DataFrame(
            {
                "WLE": est,
                "Model proportion": pp["marginal"],
                "Observed proportion": observed,
            },
            index=pd.Index(np.arange(len(lg_full)), name="Total"),
        )
        rows = {}
        for n, b in zip(names, thr):
            lgi = _esf.log_gamma([b])
            res = minimize_scalar(
                lambda t: -_esf.score_moments(lgi, [t])[1][0],
                bounds=(-20, 20),
                method="bounded",
                options=dict(xatol=1e-10),
            )
            rows[n] = {"Target": float(res.x), "Maximum information": float(-res.fun)}
        self.targeting_items = pd.DataFrame(rows).T
        return self.targeting_summary

    def global_stats_df(
        self,
        run_all_tests=False,
        test_kwargs=None,
        alpha=0.05,
        dp=3,
        filename=None,
        file_format="csv",
    ):
        """
        Build and store a summary of the whole-test results: global tests, counts of flagged items, item
        pairs and persons, and the targeting and reliability indices, in one table.

        By default the table collects what has been computed and lists the rest as 'not run', so that
        building it changes nothing else on the object. With run_all_tests=True, the analyses not yet run
        are run first, each with its own defaults unless overridden in test_kwargs: fit_statistics(
        conditional=True) (Q1 per item and overall, item-restscore, person conditional p; with bootstrap_samples given,
        the bootstrap cutoffs), res_corr_analysis() (Q3*), local_dependence_gamma(), targeting(),
        global_fit(method='andersen') for CML or 'cp-full-rcl' estimates, and, for CML estimates
        (calibrate(estimation='cml')), global_fit(method='martin-lof'). Results already present are not
        recomputed. Note that running the conditional fit replaces the person-based infit/outfit in
        item_stats and person_stats with their conditional values. With other estimates the tests that need
        them are left out, and their rows say so.

        Parameters
        ----------
        run_all_tests : bool, default False
            Run the analyses not yet run (see above), including targeting().
        test_kwargs : dict or None, default None
            Keyword arguments for the analyses run, keyed by 'fit_statistics', 'res_corr_analysis',
            'local_dependence_gamma', 'targeting', 'andersen' and 'martin-lof' (passed to
            global_fit with that method), e.g. {'fit_statistics': {'bootstrap_samples': 1000,
            'seed': 1}, 'andersen': {'split': 'exogenous', 'covariate': 'group'}}.
        alpha : float, default 0.05
            Level for the counts of flagged items, pairs and persons (BH-adjusted p-values where the
            source table has them).
        dp : int, default 3
            Decimal places.
        filename : str or None, default None
            If given, the table is also saved (filename.file_format).
        file_format : str, default 'csv'
            'csv' or 'xlsx'.

        Returns
        -------
        pandas.DataFrame
            Also stored as global_stats: one row per result -- Section, Value (the statistic, index or
            count), df, p, Details (the options defining it, or why it is not available).
        """
        kw = {k: dict(v) for k, v in (test_kwargs or {}).items()}
        cml = getattr(self, "estimation", None) == "cml"
        composite = getattr(self, "estimation", None) == "cp-full-rcl"  # Andersen's test, not Martin-Lof's
        if run_all_tests:
            if not getattr(self, "fit_conditional", False):
                self.fit_statistics(conditional=True, **kw.get("fit_statistics", {}))
            if getattr(self, "q3_star_max", None) is None:
                self.res_corr_analysis(**kw.get("res_corr_analysis", {}))
            if not hasattr(self, "local_dependence_gamma_table"):
                self.local_dependence_gamma(**kw.get("local_dependence_gamma", {}))
            if not hasattr(self, "targeting_summary"):
                self.targeting(**kw.get("targeting", {}))
            done = {k[0] for k in getattr(self, "global_fit_results", {})}
            for method in ("andersen", "martin-lof"):
                if method not in done and (cml or (method == "andersen" and composite)):
                    self.global_fit(method=method, **kw.get(method, {}))
        rows = []

        def row(section, name, value=np.nan, df=np.nan, p=np.nan, details=""):
            rows.append(
                dict(
                    Section=section,
                    Statistic=name,
                    Value=value,
                    df=df,
                    p=p,
                    Details=details,
                )
            )

        fit = "Overall fit"
        q1o = (
            getattr(self, "item_q1_overall", None)
            if getattr(self, "fit_conditional", False)
            else None
        )
        if q1o is not None:
            kind = "ability groups" if getattr(self, "_q1_ranges", None) is not None else "groups"
            for level, r in q1o.iterrows():
                row(
                    fit,
                    f"Overall Q1 ({level})",
                    r["Statistic"],
                    p=r["p"],
                    details=f"{int(r['Groups'])} {kind}; Monte Carlo p, {int(r['Samples'])} samples",
                )
        else:
            row(
                fit,
                "Overall Q1 (item scores)",
                details=(
                    "not run: fit_statistics(conditional=True)"
                    if not getattr(self, "fit_conditional", False)
                    else "not run: q1_overall_samples=None"
                ),
            )
        names = {
            "wald": "Wald test of invariance",
            "andersen": "Andersen LR",
            "martin-lof": "Martin-Lof",
        }
        results = getattr(self, "global_fit_results", {})
        for method in ("andersen", "martin-lof", "wald"):
            hits = [(k, v) for k, v in results.items() if k[0] == method]
            for (m_, det), sm in hits:
                extra = f"; covariance={sm['covariance']}" if "covariance" in sm else ""
                sparse = sm.get("score cells with < 5 persons")
                if method == "martin-lof" and sparse is not None:
                    extra += f"; {sparse:.0%} of score cells with < 5 persons"
                    if "p_method=chi2" in det and sparse > MARTIN_LOF_SPARSE:
                        extra += (
                            " (chi-square p conservative: use p_method='monte-carlo')"
                        )
                row(
                    (
                        fit
                        if method != "martin-lof"
                        else "Dimensionality and local dependence"
                    ),
                    names[method],
                    sm["statistic"],
                    sm["df"],
                    sm["p-value"],
                    det + extra,
                )
            if not hits and method != "wald":
                why = (
                    "not run: global_fit(method=%r)" % method
                    if cml or (method == "andersen" and composite)
                    else (
                        "needs calibrate(estimation='cml') or estimation='cp-full-rcl'"
                        if method == "andersen"
                        else "needs calibrate(estimation='cml')"
                    )
                )
                row(
                    (
                        fit
                        if method != "martin-lof"
                        else "Dimensionality and local dependence"
                    ),
                    names[method],
                    details=why,
                )
        dim = "Dimensionality and local dependence"
        rc = getattr(self, "residual_contrast_bootstrap", None)
        if rc is not None:
            row(
                dim,
                "First residual contrast",
                rc["first_contrast"],
                p=rc["p"],
                details=f"bootstrap 95% cutoff {rc['cutoff_95']:.{dp}f}",
            )
        else:
            row(
                dim,
                "First residual contrast",
                details="not run: fit_statistics(conditional=True, bootstrap_samples=...)",
            )
        if getattr(self, "q3_star_max", None) is not None:
            row(
                dim,
                "Q3* (largest)",
                self.q3_star_max,
                details=(
                    "conditional residuals"
                    if getattr(self, "fit_conditional", False)
                    else ""
                ),
            )
        else:
            row(dim, "Q3* (largest)", details="not run: res_corr_analysis()")
        ld = getattr(self, "local_dependence_gamma_table", None)
        if ld is not None:
            flagged = int((ld["Flagged"] & (ld["Gamma"] > 0)).sum())
            info = getattr(self, "local_dependence_gamma_info", {})
            row(
                dim,
                "Item pairs with local dependence",
                flagged,
                details=f"of {len(ld)} ordered pairs; partial gamma, BH p <= {alpha}; "
                f"p_method={info.get('p_method')}",
            )
        else:
            row(
                dim,
                "Item pairs with local dependence",
                details="not run: local_dependence_gamma()",
            )
        itf = "Item fit"
        n_items = len(self.item_names)
        if getattr(self, "fit_conditional", False):
            q1 = self.item_q1_table
            row(
                itf,
                "Items flagged by Q1",
                int((q1["p (BH)"] <= alpha).sum()),
                details=f"of {n_items}; BH p <= {alpha}",
            )
            rs = self.item_restscore_table
            row(
                itf,
                "Items flagged by item-restscore",
                int((rs["p (BH)"] <= alpha).sum()),
                details=f"of {n_items}; BH p <= {alpha}",
            )
            bt = getattr(self, "item_fit_bootstrap", None)
            if bt is not None:
                row(
                    itf,
                    "Items flagged by bootstrap infit",
                    int((bt["Infit p (Westfall-Young)"] <= alpha).sum()),
                    details=f"of {n_items}; Westfall-Young p <= {alpha}",
                )
            else:
                row(
                    itf,
                    "Items flagged by bootstrap infit",
                    details="not run: fit_statistics(conditional=True, bootstrap_samples=...)",
                )
            pc = self.person_conditional_p.dropna()
            row(
                "Person fit",
                "Persons with unexpected patterns",
                int((pc <= alpha).sum()),
                details=f"of {len(pc)} with a non-extreme total; conditional p <= {alpha} (unadjusted)",
            )
        else:
            for nm in (
                "Items flagged by Q1",
                "Items flagged by item-restscore",
                "Items flagged by bootstrap infit",
            ):
                row(itf, nm, details="not run: fit_statistics(conditional=True)")
            row(
                "Person fit",
                "Persons with unexpected patterns",
                details="not run: fit_statistics(conditional=True)",
            )
        tg = getattr(self, "targeting_summary", None)
        sec = "Targeting and reliability"
        if tg is not None:
            notes = {
                "Population mean": "normal fitted to the totals",
                "Population SD": "normal fitted to the totals",
                "Target": "location of maximum test information",
                "Test difficulty": "location with expected total half the maximum",
                "Expected |bias|": "WLE, over the population",
                "Expected SEM": "WLE, over the population",
                "Expected RMSE": "WLE, over the population",
                "rho1": "V(T) / V(R)",
                "rho2": "corr(R1, R2), two independent administrations",
                "rho3": "V(E(theta-hat | theta)) / V(theta-hat)",
                "rho4": "corr(theta-hat1, theta-hat2)",
                "rho5": "corr(theta, theta-hat)",
                "rho6": "P(R1 < R2 | theta1 < theta2)",
            }
            for k, v in tg.items():
                row(sec, k, v, details=notes.get(k, ""))
        else:
            row(
                sec, "Targeting and reliability indices", details="not run: targeting()"
            )
        tab = pd.DataFrame(rows).set_index("Statistic")
        num = ["Value", "df", "p"]
        tab[num] = tab[num].astype(float).round(dp)
        self.global_stats = tab
        if filename is not None:
            if file_format == "xlsx":
                tab.to_excel(f"{filename}.xlsx")
            else:
                tab.to_csv(f"{filename}.csv")
        return tab

    def _add_conditional_item_columns(self, stats, full, dp):
        """item_stats_df(): the conditional fit's item columns when fit_statistics(conditional=True) has
        run -- Restscore p (BH), Q1 p (BH) and, after a bootstrap, Infit/Outfit p (Westfall-Young); with
        full=True also Observed/Expected gamma, Q1 and Q1 df -- and, with full=True after targeting(),
        each item's Target and Max information."""
        idx = stats.index
        if getattr(self, "fit_conditional", False):
            rs, q1 = self.item_restscore_table, self.item_q1_table
            if full:
                stats["Observed gamma"] = (
                    rs["Observed gamma"].reindex(idx).astype(float).round(dp)
                )
                stats["Expected gamma"] = (
                    rs["Expected gamma"].reindex(idx).astype(float).round(dp)
                )
            stats["Restscore p (BH)"] = (
                rs["p (BH)"].reindex(idx).astype(float).round(dp)
            )
            if full:
                stats["Q1"] = q1["Q1"].reindex(idx).astype(float).round(dp)
                stats["Q1 df"] = q1["df"].reindex(idx)
            stats["Q1 p (BH)"] = q1["p (BH)"].reindex(idx).astype(float).round(dp)
            bt = getattr(self, "item_fit_bootstrap", None)
            if bt is not None:
                for lab in ("Infit", "Outfit"):
                    stats[f"{lab} p (Westfall-Young)"] = (
                        bt[f"{lab} p (Westfall-Young)"]
                        .reindex(idx)
                        .astype(float)
                        .round(dp)
                    )
        tgt = getattr(self, "targeting_items", None)
        if full and tgt is not None:
            stats["Target"] = tgt["Target"].reindex(idx).astype(float).round(dp)
            stats["Max information"] = (
                tgt["Maximum information"].reindex(idx).astype(float).round(dp)
            )
        return stats

    def _add_conditional_person_columns(self, stats, dp):
        """person_stats_df(): Conditional p (the probability, given the total, of a response pattern no
        more likely than the person's) when fit_statistics(conditional=True) has run."""
        if (
            getattr(self, "fit_conditional", False)
            and getattr(self, "person_conditional_p", None) is not None
        ):
            stats["Conditional p"] = (
                self.person_conditional_p.reindex(stats.index).astype(float).round(dp)
            )
        return stats

    def _icc_obs_groups(self, obs_groups):
        """icc(obs_groups=): a column of self.exogenous (str), a Series indexed by person, or labels in
        responses order -> a Series of labels indexed by person."""
        if isinstance(obs_groups, str):
            ex = getattr(self, "exogenous", None)
            if ex is None or obs_groups not in ex.columns:
                raise ValueError(
                    f"obs_groups={obs_groups!r} is not a column of this object's exogenous data."
                )
            return ex[obs_groups].reindex(self.responses.index)
        if isinstance(obs_groups, pd.Series):
            return obs_groups.reindex(self.responses.index)
        return pd.Series(list(obs_groups), index=self.responses.index)

    def _icc_overlay_points(self, items, no_of_classes, conditional, obs_groups=None):
        """
        Observed (and, with conditional=True, conditional expected) mean item scores by class interval for
        icc(cond_obs=/obs_groups=): per item, a list of (group label or None, x, observed, expected,
        half-width) arrays over class intervals. conditional=True uses the conditional item chi-square's
        grouping (_ability_groups: persons with a non-extreme total, tied locations kept together); the
        expected mean is the mean over the interval's persons of E[X_i | their total and items], and the
        half-width 1.96 SD of the interval's observed mean under the model. Otherwise the class intervals
        of the observed overlay (person-location quantiles). x is the persons' mean location.
        """
        from raschpy import _esf

        if not hasattr(self, "persons"):
            self.person_estimates(warm_corr=False)
        names = list(self.item_names)
        X = np.array(self.responses.loc[:, names], dtype=np.float64)
        loc_all = self.persons.reindex(self.responses.index).to_numpy(float)
        levels = [None]
        lab_g = None
        if obs_groups is not None:
            lab_g = self._icc_obs_groups(obs_groups)
            levels = [v for v in pd.unique(lab_g.dropna())]
        if conditional:
            mom = _esf.conditional_moments(X, self._cml_threshold_list(names))
            contributes = (np.isfinite(mom["V"]) & (mom["V"] > 0)).any(axis=1)
            classes, _ = self._ability_groups(
                np.where(contributes, loc_all, np.nan), groups=no_of_classes
            )
        out = {}
        for it in items:
            i = names.index(it)
            answered = ~np.isnan(X[:, i])
            if not conditional:
                loc = np.where(answered, loc_all, np.nan)
                q = np.nanquantile(
                    loc, [(k + 1) / no_of_classes for k in range(no_of_classes - 1)]
                )
                classes = np.where(
                    np.isnan(loc), -1, np.searchsorted(q, loc, side="right")
                )
            series = []
            for lev in levels:
                sel_g = (
                    np.ones(len(X), bool) if lev is None else (lab_g == lev).to_numpy()
                )
                xs, os_, es, hs = [], [], [], []
                for c in range(int(classes.max()) + 1):
                    sel = answered & sel_g & (classes == c)
                    if conditional:
                        sel &= np.isfinite(mom["V"][:, i]) & (mom["V"][:, i] > 0)
                    n = int(sel.sum())
                    if n == 0:
                        continue
                    xs.append(np.nanmean(loc_all[sel]))
                    os_.append(X[sel, i].mean())
                    if conditional:
                        es.append(mom["E"][sel, i].mean())
                        hs.append(1.96 * np.sqrt(mom["V"][sel, i].sum()) / n)
                series.append(
                    (lev, np.array(xs), np.array(os_), np.array(es), np.array(hs))
                )
            out[it] = series
        return out

    def _icc_overlay(
        self,
        graph,
        item,
        no_of_classes,
        cond_obs,
        cond_obs_bars,
        obs_groups,
        marker_border,
        legend_loc,
        filename,
        file_format,
        dpi,
    ):
        """Draw icc()'s observed overlay with conditional expected points (cond_obs=True: hollow diamonds,
        with 95% bars for the observed mean if cond_obs_bars) and/or split by obs_groups (single item: one
        colour and marker per group, against that group's conditional expectations), on the figure
        plot_data() returned without its observed overlay, then save it."""
        from matplotlib.lines import Line2D

        ax = graph.axes[0]
        items = item if isinstance(item, list) else [item]
        if isinstance(item, list):
            obs_groups = None  # one item only: silently ignored for several
        # the item curves span the location grid; reference lines (e.g. axhline) have 2 points
        curves = [ln for ln in ax.get_lines() if len(ln.get_xdata()) > 2]
        colours = [ln.get_color() for ln in curves[: len(items)]]
        pts = self._icc_overlay_points(items, no_of_classes, cond_obs, obs_groups)
        edge = "k" if marker_border else "none"
        markers = ["o", "s", "^", "v", "P", "X", "*", "h"]
        group_colours = [
            "#0072B2",
            "#D55E00",
            "#009E73",
            "#CC79A7",
            "#E69F00",
            "#56B4E9",
            "#F0E442",
            "#000000",
        ]
        leg = ax.get_legend()
        # legend_handles from matplotlib 3.7 (legendHandles before; removed in 3.9)
        handles = list(getattr(leg, "legend_handles", None) or leg.legendHandles) if leg else []
        labels = (
            [t.get_text() for t in ax.get_legend().get_texts()]
            if ax.get_legend()
            else []
        )
        for k, it in enumerate(items):
            for j, (lev, x, o, e, h) in enumerate(pts[it]):
                col = (
                    colours[k] if lev is None else group_colours[j % len(group_colours)]
                )
                mk = "o" if lev is None else markers[j % len(markers)]
                ax.scatter(
                    x,
                    o,
                    color=col,
                    marker=mk,
                    s=40,
                    alpha=0.7,
                    edgecolors=edge,
                    zorder=3,
                )
                if cond_obs:
                    if cond_obs_bars:
                        ax.errorbar(
                            x,
                            e,
                            yerr=h,
                            fmt="none",
                            ecolor=col,
                            elinewidth=1,
                            capsize=3,
                            alpha=0.8,
                            zorder=2,
                        )
                    ax.scatter(
                        x,
                        e,
                        marker="D",
                        s=40,
                        facecolors="none",
                        edgecolors=col,
                        linewidths=1.5,
                        zorder=4,
                    )
                if lev is not None:
                    handles.append(
                        Line2D(
                            [],
                            [],
                            color=col,
                            marker=mk,
                            linestyle="",
                            markeredgecolor=edge,
                            alpha=0.7,
                        )
                    )
                    labels.append(str(lev))
        if cond_obs:
            if (
                obs_groups is None
            ):  # with groups, their entries mark the observed points
                handles.append(
                    Line2D(
                        [],
                        [],
                        color="grey",
                        marker="o",
                        linestyle="",
                        markeredgecolor=edge,
                        alpha=0.7,
                    )
                )
                labels.append("Observed")
            handles += [
                Line2D(
                    [],
                    [],
                    color="grey",
                    marker="D",
                    linestyle="",
                    markerfacecolor="none",
                    markeredgewidth=1.5,
                ),
            ]
            labels.append("Expected given total")
        if handles:
            ax.legend(
                handles, labels, loc=legend_loc if legend_loc is not None else "best"
            )
        if filename is not None:
            graph.savefig(f"{filename}.{file_format}", dpi=dpi)
        return graph

    @staticmethod
    def _q1_overall_statistics(X, thr, labels, prep):
        """
        item_q1_overall's statistics at thresholds thr for groups labels (-1 = left out): 'item scores',
        van den Wollenberg's (1982) overall Q1, (I - 1) / I x the sum over items of each item's Q1 (the
        (I - 1) / I allows for each group's deviations summing to zero over the items, its totals being
        fixed); and, for polytomous items, 'thresholds', the same sum over each item's thresholds of the
        group's count of responses at or above the category against its conditional expectation given
        each person's total, (O - E)^2 / V, unscaled.
        """
        from raschpy import _esf

        mom = _esf.conditional_moments(X, thr, prepared=prep, probs=True)
        I = len(thr)
        out = {
            "item scores": (I - 1) / I
            * Rasch._q1_statistics(X, mom["E"], mom["V"], labels)["q1"].sum()
        }
        Ks = [len(t) for t in thr]
        if max(Ks) > 1:
            G = int(labels.max()) + 1
            total = 0.0
            for i, K in enumerate(Ks):
                Pi = mom["P"][:, i, : K + 1]
                ok = np.isfinite(Pi).all(axis=1) & ~np.isnan(X[:, i]) & (labels >= 0)
                tail = np.cumsum(Pi[ok][:, ::-1], axis=1)[:, ::-1][:, 1:]  # P(X_i >= k | total)
                hit = (X[ok, i][:, None] >= np.arange(1, K + 1)[None, :]).astype(float)
                lab = labels[ok]
                for g in range(G):
                    sel = lab == g
                    d = (hit[sel] - tail[sel]).sum(0)
                    v = (tail[sel] * (1 - tail[sel])).sum(0)
                    total += float(np.sum(np.where(v > COND_VAR_ZERO, d**2 / np.where(v > 0, v, 1.0), 0.0)))
            out["thresholds"] = total
        return out

    def _set_q1_overall(self, X, names, labels, samples, exceedances, seed, sim_thr=None):
        """
        item_q1_overall for fit_statistics(conditional=True): the overall Q1 (_q1_overall_statistics) at
        the fitted estimates, with Monte Carlo p-values: datasets simulated at sim_thr (_conditional_null;
        None: the fit) given each person's total over the items they responded to, each recalibrated with the object's calibrate()
        settings (PCM's ridge rule applied afresh), the statistics recomputed at each refit's estimates for
        the same groups; each row stops sampling once exceedances simulated statistics reach the observed
        one (Besag and Clifford's sequential p-value, exceedances / samples), else p = (1 + exceedances) /
        (1 + samples). Simulated datasets in which an item has no estimate are skipped and counted.
        """
        from raschpy import _esf

        if not samples:
            self.item_q1_overall = None
            return
        n_groups = len(np.unique(labels[labels >= 0]))
        if n_groups < 2:  # no comparison between groups: no statistic (warned in _set_q1_tables)
            self.item_q1_overall = pd.DataFrame(
                {"Statistic": np.nan, "p": np.nan, "Samples": 0, "Samples skipped": 0, "Groups": n_groups},
                index=["item scores"],
            )
            return
        thr = self._cml_threshold_list(names)
        prep = _esf.prepare(X, self._cml_max_scores(names).astype(int))
        obs = self._q1_overall_statistics(X, thr, labels, prep)
        rng = np.random.default_rng(
            [int(seed), 1] if isinstance(seed, (int, np.integer)) else seed
        )
        sim_thr = thr if sim_thr is None else sim_thr
        cml = getattr(self, "estimation", None) == "cml"
        H = self._cml_observed_hessian() if cml else None
        clone = copy.copy(self)
        if hasattr(clone, "_fixed_ridge"):
            clone.ridge = None  # each refit applies the object's ridge setting itself
        everyone = self.responses.index
        exceed = {k: 0 for k in obs}
        at = {k: None for k in obs}
        used = skipped = 0
        for _ in _progress.track(
            range(int(samples)), "Overall Q1 Monte Carlo p-value", unit="samples",
            alternative="q1_overall_samples=None leaves the overall Q1 out.",
        ):
            Y = _esf.sample_given_totals(X, sim_thr, rng)
            clone.responses = pd.DataFrame(Y, index=everyone, columns=names)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    m = clone._global_fit_subset(everyone, cml_hessian=H)
            except (ValueError, np.linalg.LinAlgError):
                skipped += 1
                continue
            if list(m.item_names) != names:
                skipped += 1  # an item without an estimate in this dataset
                continue
            used += 1
            sim = self._q1_overall_statistics(Y, m._cml_threshold_list(names), labels, prep)
            for k, T in obs.items():
                if at[k] is None:
                    exceed[k] += sim[k] >= T - 1e-9 * max(1.0, abs(T))
                    if exceedances is not None and exceed[k] >= exceedances:
                        at[k] = used
            if all(v is not None for v in at.values()):
                break
        if not used:
            warnings.warn(
                "No simulated dataset could be recalibrated with every item, so item_q1_overall has no "
                "p-value.",
                UserWarning,
                stacklevel=4,
            )
        rows = {}
        for k, T in obs.items():
            if not used:
                p = np.nan
            elif at[k] is not None:
                p = exceed[k] / at[k]
            else:
                p = (1 + exceed[k]) / (1 + used)
            rows[k] = {"Statistic": T, "p": p, "Samples": at[k] or used}
        tab = pd.DataFrame(rows).T.astype({"Statistic": float, "p": float, "Samples": int})
        tab["Samples skipped"] = skipped
        tab["Groups"] = int(labels.max()) + 1
        self.item_q1_overall = tab

    def _set_q1_tables(self, X, E, V, labels, names):
        """item_q1_table (Q1, df, p, p (BH)) and item_q1_groups (per item and group: persons, observed and
        expected mean item score, standardised difference, its variance allowing for the estimation of the
        item parameters (_threshold_influence, _corrected_group_variances)) for
        fit_statistics(conditional=True).
        """
        from raschpy import _rcl

        st = self._q1_statistics(X, E, V, labels)
        G = st["O"].shape[1]
        n_groups = len(np.unique(labels[labels >= 0]))
        if n_groups < 2:
            warnings.warn(
                f"The conditional item chi-square (Q1) compares groups of persons, but the persons with a "
                f"total that is neither the minimum nor the maximum form {n_groups} group, so item_q1_table "
                f"and item_q1_overall have no statistics. Supply q1_groups with at least 2 groups.",
                UserWarning,
                stacklevel=4,
            )
        try:
            with np.errstate(all="ignore"):
                Phi, A = self._threshold_influence(X, names)
                var = self._corrected_group_variances(
                    X, self._cml_threshold_list(names), labels, G, A, Phi
                )
        except (ValueError, np.linalg.LinAlgError, MemoryError, _rcl.SandwichTooLarge) as ex:
            var = None
            warnings.warn(
                f"item_q1_groups: the z statistics do not allow for the estimation of the item "
                f"parameters ({ex}), so their spread is below 1.",
                UserWarning,
                stacklevel=4,
            )
        df = st["df"].astype(float)
        p = np.where(df > 0, chi2_dist.sf(st["q1"], np.maximum(df, 1)), np.nan)
        q1 = np.where(df > 0, st["q1"], np.nan)  # one group: nothing compared
        tab = pd.DataFrame({"Q1": q1, "df": st["df"], "p": p}, index=names)
        tab["p (BH)"] = self._bh_correction(tab["p"].fillna(1.0)).where(
            tab["p"].notna()
        )
        self.item_q1_table = tab
        if self._q1_ranges is not None:
            glabels = list(range(1, G + 1))
        else:
            glabels = self._q1_group_names[:G]
        rows = []
        for i, item in enumerate(names):
            for g in range(G):
                n = st["n"][i, g]
                with np.errstate(invalid="ignore", divide="ignore"):
                    row = {
                        "Persons": int(n),
                        "Observed mean": st["O"][i, g] / n if n else np.nan,
                        "Expected mean": st["E"][i, g] / n if n else np.nan,
                        "z": (
                            (st["O"][i, g] - st["E"][i, g])
                            / np.sqrt(st["V"][i, g] if var is None else var[i, g])
                            if st["V"][i, g] > 0 and (var is None or var[i, g] > 1e-10)
                            else np.nan
                        ),
                    }
                if self._q1_ranges is not None:
                    row["Location from"], row["Location to"] = self._q1_ranges[g]
                rows.append(row)
        idx = pd.MultiIndex.from_product([names, glabels], names=["Item", "Group"])
        self.item_q1_groups = pd.DataFrame(rows, index=idx)
        self._q1_obs = st["q1"]

    _SHARED_THRESHOLDS = False  # RSM: one set of thresholds for every item

    def _conditional_null(self, names):
        """
        The thresholds the item-restscore draws, the conditional fit bootstrap and item_q1_overall simulate
        from: the model fitted to
        the object's responses with estimation='cp-full-rcl' and its defaults (the most accurate estimate of
        the null distribution; the statistics and refits are still the object's), or the object's fit
        if it is a 'cp-full-rcl' or 'cml' fit. If that fit fails, or does not estimate every item and
        category the object's fit does, the object's fit is used, with a UserWarning. Returns (thresholds,
        the fit used: 'cp-full-rcl' or 'own').
        """
        own = self._cml_threshold_list(names)
        if getattr(self, "estimation", None) in ("cp-full-rcl", "cml"):
            return own, "own"
        clone = copy.copy(self)
        clone._calibrate_args = dict(estimation="cp-full-rcl")
        try:
            m = clone._global_fit_subset(self.responses.index)
            thr = m._cml_threshold_list(names) if list(m.item_names) == names else None
        except (ValueError, np.linalg.LinAlgError):
            thr = None
        if thr is not None and all(
            len(a) == len(b) and np.all(np.isfinite(a)) for a, b in zip(thr, own)
        ):
            return thr, "cp-full-rcl"
        warnings.warn(
            "fit_statistics(conditional=True): the datasets for the item-restscore test, the bootstrap and the "
            "overall Q1 p-values "
            "are simulated from the object's fit, as a 'cp-full-rcl' fit of these data does not estimate "
            "every item and category it does.",
            UserWarning,
            stacklevel=4,
        )
        return own, "own"

    def _warn_conditional_unused(self, X, names):
        """
        fit_statistics(conditional=True): a UserWarning naming the score categories no person with a
        non-extreme total used (for RSM, on any item) whose fit gives them a probability above zero: their
        neighbouring thresholds are set by the smoothing or the ridge, not the data, so the conditional fit
        statistics and p-values of those items compare the data with category probabilities the data did
        not determine. Categories given probability zero (unused_categories='null'), or already named by
        the unused-category warning of the constructor (PCM) or calibrate() (RSM), which says the same, are
        not named.
        """
        thr = self._cml_threshold_list(names)
        K = self._cml_max_scores(names).astype(int)
        obs = ~np.isnan(X)
        tot = np.where(obs, X, 0).sum(1)
        Xk = X[(tot > 0) & (tot < (obs * K).sum(1))]
        finite = [bool(np.all(np.isfinite(t))) for t in thr]
        warned = getattr(self, "_unused_warned", set())  # named by an earlier warning (PCM, RSM)
        if self._SHARED_THRESHOLDS:
            if not all(finite):
                return
            cats = [k for k in range(int(K.max()) + 1) if not np.any(Xk == k) and k not in warned]
            if not cats:
                return
            what = (f"score categor{'ies' if len(cats) > 1 else 'y'} "
                    f"{', '.join(map(str, cats))} on any item")
            fix = ("collapse the unused categories, or use estimation='cml' with "
                   "unused_categories='null', which gives an unused category probability zero")
        else:
            unused = [
                (n, k)
                for i, n in enumerate(names)
                if finite[i]
                for k in range(K[i] + 1)
                if not np.any(Xk[:, i] == k) and (n, k) not in warned
            ]
            if not unused:
                return
            listed = ", ".join(f"{n} ({k})" for n, k in unused[:5])
            what = (f"{len(unused)} item score categor{'ies' if len(unused) > 1 else 'y'} "
                    f"(item (category)): {listed}{', ...' if len(unused) > 5 else ''}")
            fix = ("remove those items" if int(K.max()) == 1 else
                   "collapse the unused categories, or use estimation='cml' with "
                   "unused_categories='null', which gives an unused category probability zero")
        warnings.warn(
            f"fit_statistics(conditional=True): no person with a non-extreme total used {what}. The "
            f"thresholds next to an unused category are set by the smoothing or the ridge, not the data, so "
            f"the conditional fit statistics and p-values of the items concerned are not reliable; {fix}.",
            UserWarning,
            stacklevel=4,
        )

    @staticmethod
    def _conditional_replicate_stats(Xa, thr, prep, q1_labels=None):
        """The statistics referred to the bootstrap, for responses Xa at thresholds thr, all from the
        conditional moments: item infit and outfit; item-restscore gamma minus its expected value; Yen's
        Q3 for each item pair (correlations of the conditional residuals, upper triangle); and the first
        residual contrast (largest eigenvalue of the correlations of the standardised residuals).
        """
        from raschpy import _esf

        mom = _esf.conditional_moments(Xa, thr, prepared=prep, tables=True)
        E, V = mom["E"], mom["V"]
        valid = np.isfinite(V) & (V > 0)
        resid = np.where(valid, Xa - E, np.nan)
        r2 = np.where(valid, resid**2, 0.0)
        Vv = np.where(valid, V, 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            infit = r2.sum(0) / Vv.sum(0)
            outfit = (r2 / np.where(valid, V, 1.0)).sum(0) / valid.sum(0)
        obs_all = ~np.isnan(Xa)
        xi = np.where(obs_all, Xa, 0).astype(int)
        tot = xi.sum(1)
        gdiff = np.empty(Xa.shape[1])
        for i, tab in enumerate(mom["tables"]):
            who = obs_all[:, i]
            observed = np.zeros_like(tab)
            np.add.at(observed, (xi[who, i], tot[who] - xi[who, i]), 1.0)
            gdiff[i] = _esf.gk_gamma(observed)[0] - _esf.gk_gamma(tab)[0]
        iu = np.triu_indices(Xa.shape[1], 1)
        q3 = pd.DataFrame(resid).corr().values[iu]
        with np.errstate(invalid="ignore", divide="ignore"):
            R = pd.DataFrame(resid / np.sqrt(np.where(valid, V, np.nan))).corr().values
        R = np.where(np.isfinite(R), R, 0.0)
        eig1 = float(np.linalg.eigvalsh(R)[-1])
        out = dict(infit=infit, outfit=outfit, gdiff=gdiff, q3=q3, eig1=eig1)
        if q1_labels is not None:
            out["q1"] = Rasch._q1_statistics(Xa, E, V, q1_labels)["q1"]
        return out

    def _conditional_fit_bootstrap(self, X, names, B, seed, sim_thr=None):
        """
        Parametric bootstrap reference distributions (Johansson, 2025, 2026): B datasets simulated at
        sim_thr (_conditional_null; None: the fit) given each person's total over the items they responded to (so the score
        distribution is the observed one and no person estimates are needed), each recalibrated with the
        object's calibrate() settings (CML refits reuse the observed Hessian). For each, the conditional
        item infit and outfit, item-restscore gamma minus its expected value, Q3 for every item pair and
        the first residual contrast are computed. Each observed statistic is studentised by its simulated
        mean and SD; p-values are marginal and Westfall-Young step-down (two-sided for item fit and
        restscore, upper one-sided for Q3), with a 95% highest-density band for item fit, the 95th
        percentile as the Q3 and first-contrast cutoffs. Simulated datasets in which an item has no
        estimate are skipped and counted.
        """
        from raschpy import _esf

        rng = np.random.default_rng(seed)
        thr = self._cml_threshold_list(names)
        prep = _esf.prepare(X, self._cml_max_scores(names).astype(int))
        q1_labels = getattr(self, "_q1_labels", None)
        obs = self._conditional_replicate_stats(X, thr, prep, q1_labels)
        cml = getattr(self, "estimation", None) == "cml"
        H = self._cml_observed_hessian() if cml else None
        clone = copy.copy(self)
        everyone = self.responses.index
        sim_thr = thr if sim_thr is None else sim_thr
        sims, skipped = [], 0
        for _ in _progress.track(range(B), "Conditional fit bootstrap", unit="samples"):
            Y = _esf.sample_given_totals(X, sim_thr, rng)
            clone.responses = pd.DataFrame(Y, index=everyone, columns=names)
            try:
                m = clone._global_fit_subset(everyone, cml_hessian=H)
            except ValueError:
                skipped += 1
                continue
            if list(m.item_names) != names:
                skipped += 1  # an item without an estimate in this dataset
                continue
            sims.append(
                self._conditional_replicate_stats(
                    Y, m._cml_threshold_list(names), prep, q1_labels
                )
            )
        if not sims:
            raise ValueError(
                "No bootstrap dataset could be recalibrated with every item; the "
                "bootstrap p-values are not available."
            )
        n = len(sims)

        studentise = _esf.studentise

        def marginal(t_o, t_s, two_sided=True):
            if two_sided:
                return (1 + (np.abs(t_s) >= np.abs(t_o)).sum(0)) / (n + 1)
            return (1 + (t_s >= t_o).sum(0)) / (n + 1)

        cols = {}
        for label in ("Infit", "Outfit"):
            o = obs[label.lower()]
            sim = np.array([r[label.lower()] for r in sims])
            t_o, t_s = studentise(o, sim)
            band = np.array([_esf.hdi(sim[:, i]) for i in range(len(names))])
            cols[f"{label} MS"] = o
            cols[f"{label} 95% low"] = band[:, 0]
            cols[f"{label} 95% high"] = band[:, 1]
            cols[f"{label} p"] = marginal(t_o, t_s)
            cols[f"{label} p (Westfall-Young)"] = _esf.westfall_young(
                np.nan_to_num(t_o), np.nan_to_num(t_s)
            )
        self.item_fit_bootstrap = pd.DataFrame(cols, index=names)

        sim = np.array([r["gdiff"] for r in sims])
        t_o, t_s = studentise(obs["gdiff"], sim)
        self.item_restscore_table["p (bootstrap)"] = pd.Series(
            marginal(t_o, t_s), index=names
        )
        self.item_restscore_table["p (bootstrap, Westfall-Young)"] = pd.Series(
            _esf.westfall_young(np.nan_to_num(t_o), np.nan_to_num(t_s)), index=names
        )

        sim = np.array([r["q3"] for r in sims])
        t_o, t_s = studentise(obs["q3"], sim)
        iu = np.triu_indices(len(names), 1)
        self.q3_bootstrap = pd.DataFrame(
            {
                "Item 1": [names[i] for i in iu[0]],
                "Item 2": [names[j] for j in iu[1]],
                "Q3": obs["q3"],
                "Expected Q3": np.nanmean(sim, 0),
                "95% cutoff": np.nanpercentile(sim, 95, axis=0),
                "p": marginal(t_o, t_s, two_sided=False),
                "p (Westfall-Young)": _esf.westfall_young(
                    np.nan_to_num(t_o), np.nan_to_num(t_s), two_sided=False
                ),
            }
        ).sort_values("Q3", ascending=False, ignore_index=True)

        if q1_labels is not None:
            sim = np.array([r["q1"] for r in sims])
            t_o, t_s = studentise(obs["q1"], sim)
            self.item_q1_table["p (bootstrap)"] = pd.Series(
                marginal(t_o, t_s, two_sided=False), index=names
            )
            self.item_q1_table["p (bootstrap, Westfall-Young)"] = pd.Series(
                _esf.westfall_young(
                    np.nan_to_num(t_o), np.nan_to_num(t_s), two_sided=False
                ),
                index=names,
            )

        eig = np.array([r["eig1"] for r in sims])
        self.residual_contrast_bootstrap = dict(
            first_contrast=obs["eig1"],
            expected=float(eig.mean()),
            cutoff_95=float(np.percentile(eig, 95)),
            p=float((1 + (eig >= obs["eig1"]).sum()) / (n + 1)),
        )
        self.item_fit_bootstrap_info = dict(
            samples=n, skipped=skipped, seed=seed, null_fit=self.conditional_null_fit
        )

    @staticmethod
    def _check_cml_solver(solver, estimation):
        """calibrate(solver=): validated; applies to estimation='cml' only (else ignored, with a warning)."""
        from raschpy import _esf

        if solver not in _esf.SOLVERS:
            raise ValueError(f"solver must be one of {_esf.SOLVERS}, got {solver!r}.")
        if solver is not None and estimation != "cml":
            warnings.warn(
                f"solver={solver!r} applies to estimation='cml' only and is ignored with "
                f"estimation={estimation!r}.",
                UserWarning,
                stacklevel=3,
            )
            return None
        return solver

    def _cml_observed_hessian(self):
        """Hessian of the conditional log-likelihood at the CML estimates (computed if not kept)."""
        from raschpy import _esf

        H = getattr(self, "_cml_hessian", None)
        if H is None:
            items = list(self.item_names)
            X = np.array(self.responses.loc[:, items], dtype=np.float64)
            H = _esf.cond_loglik(X, self._cml_threshold_list(items), hessian=True)[2]
            self._cml_hessian = H
        return H

    def _cml_drop_items(self, Ks, per_item_categories=True, keep_unused=False):
        """
        Before a CML fit (SLM, PCM, RSM): remove the items whose CML estimates
        do not exist (_esf.screen_items), with one UserWarning. They go to
        self.cml_dropped_responses; persons left with no responses are added
        to self.null_persons. Returns {item: reason} for the dropped items
        (empty if none); a subclass also subsets its per-item attributes.
        With keep_unused=True (unused_categories='null'), items are not dropped for an unused category;
        self._cml_used is set to {item: boolean array over its categories, used among the persons who
        contribute}.
        """
        from raschpy import _esf

        names = list(self.item_names)
        X = np.array(self.responses.loc[:, names], dtype=np.float64)
        out = _esf.screen_items(
            X, Ks, per_item_categories=per_item_categories, keep_unused=keep_unused
        )
        keep, reasons = out[0], out[1]
        self._cml_used = (
            {names[i]: u for i, u in out[2].items()} if keep_unused else None
        )
        dropped = {names[i]: reasons[i] for i in sorted(reasons)}
        self.cml_dropped_responses = self.responses.loc[:, list(dropped)]
        if not dropped:
            return dropped
        kept = [n for n, k in zip(names, keep) if k]
        if len(kept) < 2:
            raise ValueError(
                f"CML estimates do not exist for these data: {len(kept)} item(s) would remain "
                f"after dropping {dict(list(dropped.items())[:10])}. Use the default "
                f"estimation, whose smoothing constant gives finite estimates."
            )
        self.responses = self.responses.loc[:, kept]
        empty = self.responses.index[self.responses.isnull().all(axis=1)]
        if len(empty):
            self.responses = self.responses.drop(empty)
            self.null_persons = self.null_persons.append(empty)
        self.item_names = self.responses.columns
        self.no_of_items = len(kept)
        self.person_names = self.responses.index
        self.no_of_persons = len(self.person_names)
        if getattr(self, "exogenous", None) is not None:
            self.exogenous = self.exogenous.reindex(self.person_names)
        shown = "; ".join(f"{k} ({v})" for k, v in list(dropped.items())[:10])
        if len(dropped) > 10:
            shown += f"; and {len(dropped) - 10} more"
        warnings.warn(
            f"CML estimates do not exist for {len(dropped)} item(s): {shown}. Among the persons "
            f"with a non-extreme total, an item with an unused category has no finite estimates "
            f"for the thresholds next to it, and an item that is not connected is never compared "
            f"with the others in both directions. They were left out of the CML calibration and "
            f"removed from responses (see convergence['dropped_items'] and "
            f"cml_dropped_responses). To keep them, use unused_categories='null' (PCM, RSM) or "
            f"the CML ridge (PCM's default), "
            f"collapse unused categories, or use the default estimation, whose smoothing "
            f"constant gives finite estimates.",
            UserWarning,
            stacklevel=4,
        )
        return dropped

    def _interval_stale(self, interval, attr="_se_interval"):
        """
        True when a statistics table asks for confidence bounds (interval
        not None) at a width other than that of the stored bounds, or when
        the stored standard errors have no bounds: std_errors() must then be
        run again with interval. std_errors() records the width of the
        bounds it stores in self._se_interval (MFRM: one attribute per
        sub-model, passed as attr).
        """
        return interval is not None and getattr(self, attr, None) != interval

    def _se_quantile(self, interval):
        """Multiplier of the analytic standard errors' confidence intervals, estimate +/- multiplier * SE: the
        (1 + interval) / 2 quantile of the t-distribution on persons - 1 degrees of freedom (the SEs are estimated
        from the persons' responses; as the tests on them). None if interval is None."""
        if interval is None:
            return None
        return float(t_dist.ppf(0.5 + interval / 2, max(int(self.no_of_persons) - 1, 1)))

    def _set_se_correction(self, se_correction, se_method):
        """Validate std_errors(se_correction=) and record the correction applied: se_correction for the sandwich,
        else None (the Hessian and the bootstrap take none)."""
        if se_correction not in ("kauermann-carroll", None):
            raise ValueError(f"se_correction must be 'kauermann-carroll' or None, got {se_correction!r}.")
        self.se_correction = (
            se_correction if se_method == "sandwich" else None
        )

    def _resolve_se_method(self, se_method):
        """
        SLM/PCM standard error method for the current estimates: the
        sandwich is the covariance of the Bradley-Terry estimating
        equations, which only method='log-lik' solves, so None resolves to
        'sandwich' for 'log-lik' and 'bootstrap' for 'cos'/'ls'/'evm', and
        an explicit 'sandwich' with those raises a ValueError.
        """
        est = getattr(self, "_calibrate_args", {}).get("estimation")
        if est == "cml":
            if se_method is None:
                return "sandwich"
            return se_method
        if est in ("cp-full-rcl", "cpat-rcl"):  # PCM: the composite likelihood's sandwich (no Hessian covariance)
            if se_method == "hessian":
                raise ValueError(
                    f"se_method='hessian' is the covariance of estimation='cml'; for {est!r} use "
                    "se_method='sandwich' (the default), 'bootstrap' or None."
                )
            return "sandwich" if se_method is None else se_method
        if se_method == "hessian":
            raise ValueError(
                "se_method='hessian' is the covariance of estimation='cml'; these estimates are "
                "from estimation='pair', so use se_method='sandwich', 'bootstrap' or None."
            )
        method = getattr(self, "_calibrate_args", {}).get("method") or "log-lik"
        if se_method is None:
            return "sandwich" if method == "log-lik" else "bootstrap"
        if se_method == "sandwich" and method != "log-lik":
            raise ValueError(
                f"se_method='sandwich' is the covariance of method='log-lik' "
                f"(Bradley-Terry maximum likelihood); these estimates are from "
                f"method={method!r}, so use se_method='bootstrap' (or None)."
            )
        return se_method

    @staticmethod
    def _answered_items(responses):
        """Responses restricted to the items with at least one response:
        a group of persons' own items, for the group models of dif_test(),
        drf_test() and global_fit(). An item a group did not respond to has no
        estimate in that group's calibration."""
        return responses.loc[:, responses.notna().any(axis=0).values]

    @staticmethod
    def _conditional_omnibus(pooled, focal_idx, common_items, scope="item"):
        """
        dif_test()'s omnibus test for CML estimates: a conditional likelihood ratio between nested CML fits
        to the combined reference and focal data. H0 is the pooled fit (pooled, calibrated with
        estimation='cml'); H1 adds, for the focal group, a location shift on each common item (scope='item',
        threshold shapes shared: uniform DIF) or frees all of each common item's thresholds
        (scope='full', PCM: DIF and DCF), constrained to sum to zero, since a common shift of the focal
        group's items is not identified given the persons' totals. LR = 2 (ll_1 - ll_0), df = C - 1
        ('item') or the common items' thresholds - 1 ('full'). Returns (LR, df, converged).
        """
        from raschpy import _esf, _rcl

        names = list(pooled.item_names)
        Ks = [int(k) for k in pooled._cml_max_scores(names)]
        b0 = np.concatenate(pooled._cml_threshold_list(names)).astype(float)
        A0 = pooled._cml_design(len(b0))
        X = np.array(pooled.responses.loc[:, names], dtype=np.float64)
        foc = pooled.responses.index.isin(focal_idx)
        common = [names.index(c) for c in common_items if c in names]
        off = np.concatenate([[0], np.cumsum(Ks)])
        n_new = len(common)
        Xa = np.hstack([X, np.full((len(X), n_new), np.nan)])
        Xa[np.ix_(foc, np.arange(X.shape[1], X.shape[1] + n_new))] = X[
            np.ix_(foc, common)
        ]
        Xa[np.ix_(foc, common)] = np.nan
        rows = np.concatenate([np.arange(off[c], off[c + 1]) for c in common])
        A0a, b0a = np.vstack([A0, A0[rows]]), np.concatenate([b0, b0[rows]])
        Ka = Ks + [Ks[c] for c in common]
        S = np.zeros((sum(Ka), n_new if scope == "item" else len(rows)))
        start, col = off[-1], 0
        for c in common:
            if scope == "item":
                S[start : start + Ks[c], col] = 1.0
                col += 1
            else:
                S[start : start + Ks[c], col : col + Ks[c]] = np.eye(Ks[c])
                col += Ks[c]
            start += Ks[c]
        A1 = np.hstack([A0a, S @ _rcl.zero_sum_basis(S.shape[1])])
        solver = getattr(pooled, "_cml_solver", None)
        # PCM's CML ridge: the pooled fit's penalty on the shared parameters; the focal shifts unpenalised
        Pb = getattr(pooled, "_cml_Pb", None)
        Q0 = None if Pb is None else A0.T @ Pb @ A0
        Q1 = None
        if Q0 is not None:
            Q1 = np.zeros((A1.shape[1], A1.shape[1]))
            Q1[: Q0.shape[0], : Q0.shape[0]] = Q0
        f0 = _esf.fit_cml(Xa, A0a, b0a, Ka, solver=solver, penalty=Q0)
        f1 = _esf.fit_cml(Xa, A1, b0a, Ka, solver=solver, penalty=Q1)
        LR = max(0.0, 2.0 * (f1["ll"] - f0["ll"]))
        return LR, S.shape[1] - 1, bool(f0["converged"] and f1["converged"])

    def _omnibus_h1_log_likelihood(self, group, pooled, persons, common_items):
        """
        A group model's log-likelihood at the pooled (combined-group)
        model's person locations, for dif_test()'s omnibus test (H1). When
        the groups responded to different items, the items only this group responded to are
        held at their pooled values, as under H0, so that only the shared
        items are free (df = shared items - 1), and the pooled person
        locations are shifted by the difference between the pooled and
        group models' mean locations over the shared items
        (_mean_location), since each model is centred over its items.
        """
        own = list(group.responses.columns)
        if set(own) == set(pooled.responses.columns):
            return group._log_likelihood(persons=persons)
        common = [i for i in own if i in set(common_items)]
        shift = pooled._mean_location(common) - group._mean_location(common)
        others = [i for i in own if i not in set(common)]
        scratch = group._with_pooled_items(pooled, others, shift)
        return scratch._log_likelihood(persons=persons - shift)

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
            raise ValueError(
                f"reference='{reference}' is not a value of '{covariate}'."
            )

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

    def _dif_linked_focal_levels(
        self,
        cov_values,
        reference,
        focal_levels,
        test,
        need_items=True,
        need_facets=0,
        min_persons=1,
    ):
        """
        The focal groups that can be compared with the reference group in
        dif_test() / drf_test(): those that share at least one item with
        it (need_items) and, for MFRM, at least `need_facets` facet elements
        that scored at least `min_persons` persons in each. With more than
        two groups the data can be connected through other groups while a
        focal group shares none of these with the reference; such
        groups are left out, with one warning, and a ValueError is raised
        if no focal group is left.

        Returns (focal levels kept, {left-out focal level: reason}).
        """
        resp = self.responses
        mfrm = isinstance(resp.index, pd.MultiIndex)
        persons = resp.index.get_level_values(1) if mfrm else resp.index

        def _items(idx):
            sub = resp[persons.isin(idx)]
            return set(resp.columns[sub.notna().any(axis=0).values])

        def _facets(idx):
            sub = resp[persons.isin(idx)]
            counts = sub.notna().any(axis=1).groupby(level=0).sum()
            return set(counts.index[counts >= min_persons])

        ref_idx = cov_values.index[cov_values == reference]
        ref_items = _items(ref_idx) if need_items else None
        ref_facets = _facets(ref_idx) if need_facets else None
        reasons = {}
        for focal in focal_levels:
            idx = cov_values.index[cov_values == focal]
            why = []
            if need_items and not ref_items & _items(idx):
                why.append("no shared items")
            if need_facets and len(ref_facets & _facets(idx)) < need_facets:
                why.append(
                    (
                        f"no shared {self.facets_label}"
                        if need_facets == 1
                        else f"fewer than {need_facets} shared {self.facets_label}"
                    )
                    + (
                        f" (each scoring at least {min_persons} persons in both)"
                        if min_persons > 1
                        else ""
                    )
                )
            if why:
                reasons[focal] = " and ".join(why)
        if not reasons:
            return focal_levels, {}
        kept = [f for f in focal_levels if f not in reasons]
        detail = "; ".join(f"{f!r}: {r}" for f, r in reasons.items())
        if not kept:
            raise ValueError(
                f"No focal group can be compared with the reference group "
                f"{reference!r} in {test}() ({detail})."
            )
        warnings.warn(
            f"{len(reasons)} focal group(s) left out of {test}(), as they cannot "
            f"be compared with the reference group {reference!r} ({detail}); the "
            f"data link them only through other groups. Choose a different "
            f"reference= to compare them.",
            UserWarning,
            stacklevel=3,
        )
        return kept, reasons

    @staticmethod
    def _nan_omnibus_rows(rows, excluded):
        """Add a row of NaN, Flagged=False, to an omnibus table's rows for
        each focal group left out (_dif_linked_focal_levels)."""
        for focal in excluded:
            rows[focal] = {"LR": np.nan, "df": np.nan, "p": np.nan, "Flagged": False}
        return rows

    def _dif_category_column(
        self,
        diff,
        se,
        p,
        common_items,
        welch,
        welch_df,
        category_thresholds,
        category_alpha,
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

    def _dif_lr_table(
        self, lr_rows, common_items, correction, alpha, diff, logit_threshold
    ):
        """
        Post-processing shared by every class's per-item DIF LR test
        (test='lr'/'both'): multiple-comparison correction and the
        Flagged_LR gate. The LR statistics themselves (lr_rows, a dict
        {item: {'LR':..., 'p_LR':...}}) are computed by the caller, since
        that part IS genuinely model-specific (each class's scratch-
        model reconstruction for H0).

        Returns
        -------
        pandas.DataFrame, index=common_items, columns 'LR', 'p_LR',
        'p_LR (corrected)', 'Flagged_LR'.
        """
        lr_table = pd.DataFrame(lr_rows).T.loc[common_items]
        if correction == "bonferroni":
            lr_table["p_LR (corrected)"] = (lr_table["p_LR"] * len(lr_table)).clip(
                upper=1.0
            )
        elif correction == "bh":
            lr_table["p_LR (corrected)"] = self._bh_correction(lr_table["p_LR"])
        else:
            lr_table["p_LR (corrected)"] = lr_table["p_LR"]
        lr_table["Flagged_LR"] = (lr_table["p_LR (corrected)"] <= alpha) & (
            diff.abs() >= logit_threshold
        )
        return lr_table

    def _dif_validate(
        self,
        correction,
        selection_method,
        test,
        category_thresholds,
        reference_n,
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
        if (
            len(category_thresholds) != 2
            or category_thresholds[0] >= category_thresholds[1]
        ):
            raise ValueError(
                "category_thresholds must be (b_threshold, c_threshold) with "
                "b_threshold < c_threshold."
            )
        if reference_n <= 1:
            raise ValueError("reference_n must be greater than 1.")
        if dcf_cap is not None and dcf_cap <= 0:
            raise ValueError("dcf_cap must be greater than 0.")

    def _dif_item_wald_block(
        self,
        ref_model,
        focal_model,
        ref_boot,
        focal_boot,
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
        ref_cov=None,
        focal_cov=None,
    ):
        """
        Shared item-location purification + per-item Wald test block from
        dif_test() -- byte-identical across SLM/PCM/RSM/MFRM aside from
        which bootstrap matrix the caller supplies for tc's variance
        (item_bootstrap for SLM/RSM, central_bootstrap for PCM, a matrix
        MFRM builds per-model via _bootstrap_value_matrix before calling
        this, since MFRM stores its bootstrap resamples per facet-element-
        parameterisation rather than under one fixed attribute name).
        Only the item-location component; each class's threshold-
        structure DIF/DCF block (PCM's per-item, RSM/MFRM's shared-vector
        category-width test) is NOT part of this -- those differ by
        design (see calibrate_anchor's docstring on why PCM/RSM's
        threshold handling isn't the same statistical procedure) and stay
        in each subclass.

        Returns
        -------
        dict with keys: common_items, tc, focal_shifted, diff, se, z, p,
        p_corrected, flagged, selected, anchor_table, welch_df
        (pandas.Series or None), category_col (pandas.Series or None).
        """
        common_items = ref_model.items.index.intersection(focal_model.items.index)
        if len(common_items) == 0:
            # possible with more than two groups: the data can be connected
            # through another group
            raise ValueError(
                "The reference and focal group share no items, so there "
                "is nothing to compare."
            )
        if len(common_items) == 1:
            dropped = {
                lab: (getattr(mod, "convergence", None) or {}).get("dropped_items")
                for lab, mod in (("reference", ref_model), ("focal", focal_model))
            }
            dropped = {k: v for k, v in dropped.items() if v}
            why = (
                " Items without CML estimates were removed from the group fits ("
                + "; ".join(
                    f"{lab} group: " + ", ".join(f"{i} ({r})" for i, r in v.items())
                    for lab, v in dropped.items()
                )
                + "); collapsing their unused categories, or an estimation method with "
                "smoothing (the default estimation), keeps them."
                if dropped
                else ""
            )
            raise ValueError(
                f"The reference and focal group share only one item ({common_items[0]}), and DIF "
                f"needs at least two: an item's DIF is measured against the other common items."
                + why
            )
        ref_items = ref_model.items.loc[common_items]
        focal_items = focal_model.items.loc[common_items]
        ref_se_item = ref_model.item_se.loc[common_items]
        focal_se_item = focal_model.item_se.loc[common_items]

        # Leave-one-out per item, with tc's own variance -- see
        # _purification_loo_tc.
        tc, ref_var, focal_var, selected, anchor_table = self._purification_loo_tc(
            ref_items,
            focal_items,
            ref_se_item,
            focal_se_item,
            selection_method,
            corr_tol,
            sd_ratio_tol,
            min_anchors,
            wald_alpha,
            ref_boot=ref_boot,
            focal_boot=focal_boot,
            ref_cov=ref_cov,
            focal_cov=focal_cov,
            n_ref=ref_model.no_of_persons,
            n_focal=focal_model.no_of_persons,
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
                diff.values,
                ref_se_eff.values,
                ref_n_item,
                focal_se_eff.values,
                focal_n_item,
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

        flagged = (p_corrected <= alpha) & (diff.abs() >= logit_threshold)

        category_col = None
        if category:
            category_col = self._dif_category_column(
                diff,
                se,
                p,
                common_items,
                welch,
                welch_df,
                category_thresholds,
                category_alpha,
            )

        n_kept, n_common = len(selected), len(common_items)
        if selection_method != "none" and n_kept < n_common / 2:
            warnings.warn(
                f"The anchor selection kept only {n_kept} of the {n_common} items shared by the reference and a focal "
                f"group: DIF this widespread leaves too few items to "
                f"place the groups on a common scale, so the anchors may themselves carry DIF and the item-level "
                f"results depend on which items are kept. Check whether the items measure the same thing in both "
                f"groups before interpreting the DIF table.",
                UserWarning,
                stacklevel=3,
            )

        return {
            "common_items": common_items,
            "tc": tc,
            "focal_shifted": focal_shifted,
            "diff": diff,
            "se": se,
            "z": z,
            "p": p,
            "p_corrected": p_corrected,
            "flagged": flagged,
            "selected": selected,
            "anchor_table": anchor_table,
            "welch_df": welch_df,
            "category_col": category_col,
        }

    def _dif_item_table(
        self,
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
        welch_df=None,
        category_col=None,
        lr_table=None,
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
        if welch_df is not None:  # Welch t on Welch-Satterthwaite df
            table = table.rename(columns={"z": "t"})
            table["df"] = welch_df
        if category_col is not None:
            table["Category"] = category_col
        if lr_table is not None:
            table = table.join(lr_table)
        return table

    def _parameter_recovery_table(
        self,
        generating,
        estimated,
        se=None,
        alpha=0.05,
        correction="bh",
        groups=None,
        divergence_tol=0.1,
        df=None,
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
            z (or t)/p/Flagged columns (left all-NaN/False) -- used for
            parameter types with no available SE.
        df : float or None, default None
            Degrees of freedom of a t reference for error / SE (persons - 1
            for item-side parameters, whose SEs are estimated from the
            persons' responses): the statistic column is then 't', followed
            by a 'df' column. None: a normal reference, column 'z' (person
            rows).
        alpha : float, default 0.05
            Significance threshold for Flagged/Flagged_corrected.
        correction : {'bh', 'bonferroni', None}, default 'bh'
            Multiple-comparison correction for p_corrected, matching
            dif_test's convention (self._bh_correction for BH; the
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
            z (or t and df when df is given), p, p_corrected, Flagged,
            Flagged_corrected, and Group if groups was given.
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
            if df is None:
                p = pd.Series(2 * norm.sf(z.abs()), index=idx)
            else:
                p = pd.Series(2 * t_dist.sf(z.abs(), df), index=idx)
        else:
            z = pd.Series(np.nan, index=idx)
            p = pd.Series(np.nan, index=idx)

        if correction == "bh":
            valid_p = p.dropna()
            p_corrected = (
                self._bh_correction(valid_p).reindex(idx) if len(valid_p) else p.copy()
            )
        elif correction == "bonferroni":
            n_valid = p.notna().sum()
            p_corrected = (p * n_valid).clip(upper=1.0)
        else:
            p_corrected = pd.Series(np.nan, index=idx)

        flagged = p <= alpha
        flagged_corrected = p_corrected <= alpha

        table = pd.DataFrame(
            {
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
            }
        )
        if df is not None:
            table = table.rename(columns={"z": "t"})
            table.insert(table.columns.get_loc("t") + 1, "df", float(df))
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
                r_squared = r**2
                sd_ratio = float(np.std(e) / np.std(g))
                ols_slope, ols_intercept = (float(v) for v in np.polyfit(g, e, 1))
                hr = HuberRegressor(epsilon=1.5, max_iter=200).fit(g.reshape(-1, 1), e)
                huber_slope = float(hr.coef_[0])
                huber_intercept = float(hr.intercept_)
                divergence_flag = bool(abs(ols_slope - huber_slope) > divergence_tol)

            return pd.Series(
                {
                    "RMSE": rmse,
                    "MAE": mae,
                    "bias": bias,
                    "r": r,
                    "r_squared": r_squared,
                    "SD_ratio": sd_ratio,
                    "OLS_slope": ols_slope,
                    "OLS_intercept": ols_intercept,
                    "Huber_slope": huber_slope,
                    "Huber_intercept": huber_intercept,
                    "Divergence_flag": divergence_flag,
                    "Prop_flagged": prop_flagged,
                }
            )

        summary = {"Overall": _summary_slice(table)}
        if groups is not None:
            for g_val in sorted(table["Group"].dropna().unique(), key=str):
                summary[str(g_val)] = _summary_slice(table[table["Group"] == g_val])
        summary = pd.DataFrame(summary)

        return table, summary

    def _parameter_estimation_residual(
        self,
        true_theta,
        exp_score_generating,
        exp_score_estimated,
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

        Only evaluated over self.responses' observed (non-null)
        cells, so missing-data conditions are handled automatically.

        Parameters
        ----------
        true_theta : pandas.Series
            TRUE generating person locations, indexed by person -- NOT
            this object's recovered or pseudo-estimated persons.
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
        self,
        generating,
        estimated,
        label,
        filename=None,
        file_format="png",
        dpi=300,
        plot_style="white",
        font="Times New Roman",
        title_font_size=15,
        axis_font_size=12,
        labelsize=12,
        figsize=(8, 6),
        colors=None,
        markers=None,
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
                g,
                e,
                alpha=0.6,
                s=30,
                color=color,
                marker=marker,
                edgecolors="k",
                linewidths=0.3,
                label=str(glabel),
                zorder=3,
            )
            all_vals.append(g)
            all_vals.append(e)
            # fit lines need generating values that vary (e.g. not neutral anchors, all generated at 0)
            spread = len(np.unique(g)) >= 2
            if len(g) >= 2 and spread:
                m, b = np.polyfit(g, e, 1)
                xseq = np.linspace(g.min(), g.max(), 50)
                ax.plot(xseq, m * xseq + b, color=color, linestyle="--", lw=1.2)
            if len(g) >= 3 and spread:
                hr = HuberRegressor(epsilon=1.5, max_iter=200).fit(g.reshape(-1, 1), e)
                xseq = np.linspace(g.min(), g.max(), 50)
                ax.plot(
                    xseq,
                    hr.coef_[0] * xseq + hr.intercept_,
                    color=color,
                    linestyle=":",
                    lw=1.2,
                )

        if all_vals:
            combined = np.concatenate(all_vals)
            lo, hi = float(combined.min()), float(combined.max())
            xseq = np.linspace(lo, hi, 100)
            ax.plot(xseq, xseq, color="black", lw=1, label="Identity", zorder=2)

        ax.set_xlabel(
            f"Generating {label}",
            fontsize=axis_font_size,
            fontweight="bold",
            fontname=font,
        )
        ax.set_ylabel(
            f"Estimated {label}",
            fontsize=axis_font_size,
            fontweight="bold",
            fontname=font,
        )
        ax.set_title(
            f"{label} recovery",
            fontsize=title_font_size,
            fontweight="bold",
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
        model's recovery_stats() docstring). MFRM's version of this
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
            probe = PCM(
                self.responses, max_score_vector=self.generating.max_score_vector
            )
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

    _GLOBAL_FIT_METHODS = ("wald", "andersen", "martin-lof")
    _GLOBAL_FIT_SPLITS = ("score", "score_groups", "person_location", "random", "exogenous")

    def _global_fit_groups(self, split, covariate, seed, split_prop=0.5, max_groups=2):
        """
        Person groups (index objects) for global_fit(): two groups, or with split='exogenous' up to
        max_groups (one per covariate value).
        """
        resp = self.responses
        if split == "score_groups":
            return self._score_group_split(resp)
        if split in ("score", "person_location"):
            if split == "score":
                value = resp.mean(
                    axis=1
                )  # mean observed score: comparable under missing data
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
            raise ValueError(
                "split='exogenous' requires exogenous data and covariate=."
            )
        values = self.exogenous[covariate].reindex(resp.index)
        levels = sorted(values.dropna().unique(), key=str)
        if len(levels) < 2 or len(levels) > max_groups:
            need = "exactly two values" if max_groups == 2 else "at least two values"
            raise ValueError(
                f"split='exogenous' needs a covariate with {need}; "
                f"{covariate!r} has {len(levels)}."
                + (
                    " method='andersen' takes more than two."
                    if max_groups == 2 and len(levels) > 2
                    else ""
                )
            )
        return {str(lv): resp.index[(values == lv).values] for lv in levels}

    def _score_group_split(self, resp):
        """
        split='score_groups': two groups of whole score groups (persons with the same score are never
        separated), cut where the persons with non-extreme scores divide most evenly, as in iarm's
        score_groups (Mueller, 2020); on a tie between two cuts, the higher one if the lowest score is no
        more frequent than the highest, else the lower. Persons with extreme scores join the group at
        their end. The score is the mean observed score, which with complete data orders persons as the
        total does.
        """
        X = resp.to_numpy(dtype=float)
        obs = ~np.isnan(X)
        maxes = np.nanmax(X, axis=0)
        mins = np.nanmin(X, axis=0)
        total = np.where(obs, X, 0).sum(1)
        top = obs @ np.nan_to_num(maxes)
        bottom = obs @ np.nan_to_num(mins)
        answered = obs.any(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            value = np.where(answered, total / np.maximum(obs.sum(1), 1), np.nan)
        extreme = (total <= bottom) | (total >= top)
        inner = answered & ~extreme
        levels, counts = np.unique(np.round(value[inner], 12), return_counts=True)
        if len(levels) < 2:
            raise ValueError(
                "split='score_groups' needs at least two different non-extreme scores."
            )
        dist = np.abs(np.cumsum(counts) - counts.sum() / 2)[:-1]
        best = np.flatnonzero(np.isclose(dist, dist.min()))
        if len(best) == 2:
            vals, freq = np.unique(np.round(value[answered], 12), return_counts=True)
            best = best[1] if freq[0] <= freq[-1] else best[0]
        else:
            best = best[0]
        cut = levels[best]
        low = answered & (np.round(value, 12) <= cut)
        high = answered & (np.round(value, 12) > cut)
        return {"low": resp.index[low], "high": resp.index[high]}

    @staticmethod
    def _martin_lof_stat(resp, A, fA, fB, f):
        """Martin-Lof LR and df from the subset and joint CML fits and the responses (A + B columns)."""
        X = resp.to_numpy(dtype=float)
        obs = ~np.isnan(X)
        nA = len(A)
        maxes = f._cml_max_scores(list(resp.columns))
        Xi = np.where(obs, X, 0)
        r, s_ = Xi[:, :nA].sum(1), Xi[:, nA:].sum(1)
        MA, MB = obs[:, :nA] @ maxes[:nA], obs[:, nA:] @ maxes[nA:]
        keys = np.unique(obs, axis=0, return_inverse=True)[1].ravel()
        sat1 = sat0 = 0.0
        df_scores = 0
        sparse = cells = 0
        for key in np.unique(keys):
            g = keys == key
            Ng = g.sum()
            _, n_rs = np.unique(np.c_[r[g], s_[g]], axis=0, return_counts=True)
            n_cells = int((MA[g][0] + 1) * (MB[g][0] + 1))
            cells += n_cells
            sparse += n_cells - int(
                (n_rs >= 5).sum()
            )  # cells with fewer than 5 persons
            _, n_t = np.unique(r[g] + s_[g], return_counts=True)
            sat1 += float((n_rs * np.log(n_rs / Ng)).sum())
            sat0 += float((n_t * np.log(n_t / Ng)).sum())
            df_scores += int(MA[g][0] * MB[g][0])
        ll_A, ll_B, ll = (m.convergence["log_likelihood"] for m in (fA, fB, f))
        LR = max(2.0 * (ll_A + ll_B - ll + sat1 - sat0), 0.0)
        df = (
            df_scores
            + fA._cml_free_parameters()
            + fB._cml_free_parameters()
            - f._cml_free_parameters()
        )
        return LR, df, (ll_A, ll_B, ll), int(len(np.unique(keys))), sparse / cells

    def _martin_lof(
        self,
        split,
        alpha,
        p_method="chi2",
        no_of_samples=500,
        seed=None,
        mc_exceedances=None,
    ):
        """
        global_fit(method='martin-lof'): Martin-Lof's test of unidimensionality
        for two item subsets A and B,
        LR = 2 (ll_A + ll_B - ll + sum n_rs log(n_rs / N) - sum n_t log(n_t / N)),
        with ll the CML log-likelihoods of A, B and all items, n_rs the persons
        with subtotals r on A and s on B and n_t those with total t, both
        within each missingness pattern (N its persons);
        df = sum over patterns of M_A M_B, plus q_A + q_B - q (free parameters).
        Items without CML estimates in their subset's fit or the joint fit are
        left out (repeated until stable).
        """
        names = list(self.item_names)
        if isinstance(split, str):
            if split != "median":
                raise ValueError(
                    f"For method='martin-lof', split must be 'median' or a list of the "
                    f"items in the first subset, got {split!r}."
                )
            order = self._item_locations_for_split().reindex(names).rank(method="first")
            A = [n for n in names if order[n] <= len(names) // 2]
            label = "median"
        else:
            A = [n for n in names if n in set(split)]
            unknown = [n for n in split if n not in set(names)]
            if unknown:
                raise ValueError(f"Items not in the data: {unknown[:10]}.")
            label = "items"
        B = [n for n in names if n not in set(A)]
        if len(A) < 2 or len(B) < 2:
            raise ValueError(
                "method='martin-lof' needs at least two items in each subset."
            )
        everyone = self.responses.index
        while True:
            try:
                fA = self._global_fit_subset(everyone, items=A)
                fB = self._global_fit_subset(everyone, items=B)
                f = self._global_fit_subset(everyone, items=A + B)
            except ValueError as e:
                if str(e).startswith("CML estimates do not exist"):
                    reason = str(e).split(": ", 1)[-1].split(". ")[0]
                    raise ValueError(
                        f"method='martin-lof': CML estimates do not exist for one item "
                        f"subset ({reason})."
                    ) from None
                raise
            kA = [n for n in A if n in fA.item_names and n in f.item_names]
            kB = [n for n in B if n in fB.item_names and n in f.item_names]
            if kA == A and kB == B:
                break
            A, B = kA, kB
            if len(A) < 2 or len(B) < 2:
                raise ValueError(
                    "method='martin-lof': fewer than two items with CML estimates "
                    "remain in one subset."
                )
        resp = self.responses.loc[:, A + B]
        LR, df, (ll_A, ll_B, ll), n_patterns, sparse = self._martin_lof_stat(
            resp, A, fA, fB, f
        )
        p_chi2 = float(chi2_dist.sf(LR, df))
        p, used, skipped, stopped = p_chi2, None, None, None
        if p_method == "monte-carlo":
            # simulate from the joint fit given each person's total and observed items (the
            # conditional distribution does not involve the persons), refit, recompute the LR
            from raschpy import _esf

            rng = np.random.default_rng(seed)
            thr = f._cml_threshold_list(A + B)
            clone = copy.copy(self)
            exceed, used, skipped, stopped = 0, 0, 0, False

            def observed_hessian(m):
                # the replicates' fits use it as a fixed Newton matrix (chord method): same design,
                # nearby estimates, so a few gradient evaluations replace a Hessian per step
                H = getattr(m, "_cml_hessian", None)
                if H is None:
                    items = list(m.item_names)
                    Xm = np.array(m.responses.loc[:, items], dtype=np.float64)
                    H = _esf.cond_loglik(
                        Xm, m._cml_threshold_list(items), hessian=True
                    )[2]
                return H

            H_A, H_B, H_all = (observed_hessian(m) for m in (fA, fB, f))
            for _ in _progress.track(range(no_of_samples), "Martin-Lof Monte Carlo p-value", unit="samples",
                                     alternative="p_method='chi2' takes seconds."):
                sim = _esf.sample_given_totals(resp.to_numpy(dtype=float), thr, rng)
                clone.responses = pd.DataFrame(
                    sim, index=resp.index, columns=resp.columns
                )
                try:
                    gA = clone._global_fit_subset(everyone, items=A, cml_hessian=H_A)
                    gB = clone._global_fit_subset(everyone, items=B, cml_hessian=H_B)
                    g = clone._global_fit_subset(
                        everyone, items=A + B, cml_hessian=H_all
                    )
                except ValueError:
                    skipped += 1
                    continue
                if (
                    len(gA.item_names) < len(A)
                    or len(gB.item_names) < len(B)
                    or len(g.item_names) < len(A) + len(B)
                ):
                    skipped += 1  # an item without CML estimates: not comparable
                    continue
                exceed += self._martin_lof_stat(clone.responses, A, gA, gB, g)[0] >= LR
                used += 1
                if mc_exceedances is not None and exceed >= mc_exceedances:
                    stopped = True  # Besag-Clifford sequential stopping
                    break
            if not used:
                raise ValueError(
                    "method='martin-lof': no Monte Carlo sample had CML estimates for every item."
                )
            p = exceed / used if stopped else (1 + exceed) / (1 + used)
        excluded = [n for n in names if n not in set(A + B)]
        self.global_fit_table = None
        self.global_fit_excluded = excluded
        self.global_fit_groups = {"A": fA, "B": fB, "all": f}
        self.global_fit_summary = pd.Series(
            {
                "method": "martin-lof",
                "split": label,
                "statistic": LR,
                "df": df,
                "subsets": {"A": A, "B": B},
                "log-likelihoods": {"A": ll_A, "B": ll_B, "all": ll},
                "p_method": p_method,
                "p-value": p,
                "p-value (chi-square)": p_chi2,
                "Monte Carlo samples": used,
                "Monte Carlo samples skipped": skipped,
                "Monte Carlo stopped early": stopped,
                "reject": bool(p <= alpha),
                "missingness patterns": n_patterns,
                "score cells with < 5 persons": sparse,
                "items excluded": len(excluded),
            },
            name="Global fit",
        )
        if p_method == "chi2" and sparse > MARTIN_LOF_SPARSE:
            warnings.warn(
                f"{sparse:.0%} of the Martin-Lof score table's cells (subtotal on subset A by "
                f"subtotal on subset B) have fewer than 5 persons, so the chi-square p-value is "
                f"likely conservative (in null simulations, rejection rates fell towards 0 at "
                f"alpha = 0.05 as this share grew). For a test with a nominal false-positive "
                f"rate, re-run with global_fit(method='martin-lof', p_method='monte-carlo').",
                UserWarning,
                stacklevel=3,
            )
        if excluded:
            warnings.warn(
                f"{len(excluded)} item(s) left out of the Martin-Lof test because they "
                f"have no CML estimates in their subset or in all items: "
                f"{excluded[:10]}{' ...' if len(excluded) > 10 else ''}. See "
                f"global_fit_excluded.",
                UserWarning,
                stacklevel=3,
            )
        return self.global_fit_summary

    def _andersen_lr(
        self,
        split,
        covariate,
        seed,
        split_prop,
        alpha,
        p_method="chi2",
        no_of_samples=500,
        mc_exceedances=None,
    ):
        """
        global_fit(method='andersen'): Andersen's conditional likelihood
        ratio test, LR = 2 (sum_g ll_g - ll_pooled), each conditional
        log-likelihood at its CML maximum, with df = (groups - 1) x the
        model's free parameters. All fits use the items with CML estimates
        in both groups (repeated until stable, since dropping an item changes
        which persons contribute); the pooled model is fitted to the persons
        of the two groups only.
        """
        groups = self._global_fit_groups(split, covariate, seed, split_prop, max_groups=np.inf)
        names = list(groups)
        common = [
            c
            for c in self.item_names
            if all(self.responses.loc[groups[n], c].notna().any() for n in names)
        ]
        fitted = {}
        while True:
            if len(common) < 2:
                break  # reported below
            try:
                for n in names:
                    fitted[n] = self._global_fit_subset(groups[n], items=common)
            except ValueError as e:
                if str(e).startswith("CML estimates do not exist"):
                    reason = str(e).split(": ", 1)[-1].split(". ")[0]
                    raise ValueError(
                        f"method='andersen': CML estimates do not exist in one group "
                        f"({reason}). Use unused_categories='null', which keeps items with a "
                        f"category unused in a group, or method='wald', which leaves out the "
                        f"parameters not estimable in both groups."
                    ) from None
                raise
            kept = [c for c in common if all(c in fitted[n].item_names for n in names)]
            if kept == common:
                break
            common = kept
        if len(common) < 2:
            raise ValueError(
                "method='andersen': fewer than two items have CML estimates in both "
                "groups, so there is nothing to compare."
            )
        both = groups[names[0]].append([groups[n] for n in names[1:]])
        pooled = self._global_fit_subset(both, items=common)
        self._andersen_item_tables(fitted, pooled, groups, names)
        ll_groups = {n: fitted[n].convergence["log_likelihood"] for n in names}
        ll0 = pooled.convergence["log_likelihood"]
        LR = max(2.0 * (sum(ll_groups.values()) - ll0), 0.0)
        df = (
            sum(fitted[n]._cml_free_parameters() for n in names)
            - pooled._cml_free_parameters()
        )
        p_chi2 = float(chi2_dist.sf(LR, df))
        p, used, skipped, stopped = p_chi2, None, None, None
        if p_method == "auto":
            # RSM with p_method=None: a fit that needed the unused-category width ridge makes the chi-square
            # p-value conservative, so Monte Carlo then
            ridge_used = any(
                getattr(m, "_cml_penalty", None) is not None for m in [*fitted.values(), pooled]
            )
            p_method = "monte-carlo" if ridge_used else "chi2"
            if ridge_used and mc_exceedances is None:
                mc_exceedances = 10
        if p_method == "monte-carlo":
            p, used, skipped, stopped = self._andersen_monte_carlo(
                LR, groups, names, common, both, fitted, pooled,
                no_of_samples, seed, mc_exceedances,
            )
        excluded = [c for c in self.item_names if c not in common]
        self.global_fit_table = None
        self.global_fit_excluded = excluded
        self.global_fit_groups = fitted
        summary = {"method": "andersen", "split": split, "statistic": LR, "df": df}
        if split in ("score", "person_location"):
            summary["split_prop"] = split_prop
        if split == "exogenous":
            summary["covariate"] = covariate
        summary.update(
            {
                "log-likelihoods": {**ll_groups, "pooled": ll0},
                "p_method": p_method,
                "p-value": p,
                "p-value (chi-square)": p_chi2,
                "Monte Carlo samples": used,
                "Monte Carlo samples skipped": skipped,
                "Monte Carlo stopped early": stopped,
                "reject": bool(p <= alpha),
                "group sizes": {n: len(groups[n]) for n in names},
                "items excluded": len(excluded),
            }
        )
        self.global_fit_summary = pd.Series(summary, name="Global fit")
        not_converged = [
            n
            for n, m in {**fitted, "pooled": pooled}.items()
            if not m.convergence["converged"]
        ]
        if not_converged:
            warnings.warn(
                f"The CML fit did not converge for {not_converged}; the likelihood ratio "
                f"may be unreliable.",
                UserWarning,
                stacklevel=3,
            )
        if p_method == "chi2" and any(
            getattr(m, "_cml_marked", None) is not None
            for m in list(fitted.values()) + [pooled]
        ):
            warnings.warn(
                "unused_categories='null' kept categories unused in a group, so their "
                "thresholds were left out of that group's fit; the chi-square p-value is then "
                "liberal (in null simulations of PCM with sparse categories it rejected 7-25% "
                "of the time at alpha = 0.05). For a calibrated test, re-run with "
                "p_method='monte-carlo'.",
                UserWarning,
                stacklevel=3,
            )
        if excluded:
            warnings.warn(
                f"{len(excluded)} item(s) left out of the likelihood ratio test because "
                f"they have no CML estimates in one group (not responded to, an unused "
                f"category, or not compared with the other items in both directions): "
                f"{excluded[:10]}{' ...' if len(excluded) > 10 else ''}. See "
                f"global_fit_excluded.",
                UserWarning,
                stacklevel=3,
            )
        return self.global_fit_summary

    def _andersen_monte_carlo(
        self, LR, groups, names, items, both, fitted, pooled, no_of_samples, seed, mc_exceedances
    ):
        """
        global_fit(method='andersen', p_method='monte-carlo'): datasets simulated from the pooled CML
        fit given each person's total and the items they responded to (the conditional distribution does
        not involve the persons), each refitted by group and pooled with the same settings; p = (1 + the
        number of simulated LRs at least the observed) / (1 + samples used). The groups are those of the
        observed data (a score split's groups are unchanged, the totals being fixed). A sample is skipped
        if a fit leaves out an item. Returns (p, used, skipped, stopped early).
        """
        from raschpy import _esf

        rng = np.random.default_rng(seed)
        resp = self.responses.loc[both, items]
        thr = pooled._cml_threshold_list(items)
        clone = copy.copy(self)
        hess = {n: getattr(fitted[n], "_cml_hessian", None) for n in names}
        h0 = getattr(pooled, "_cml_hessian", None)
        exceed, used, skipped, stopped = 0, 0, 0, False
        for _ in _progress.track(range(no_of_samples), "Andersen LR Monte Carlo p-value", unit="samples",
                                 alternative="mc_exceedances= stops early once the p-value is clearly large; "
                                 "p_method='chi2' takes seconds."):
            sim = _esf.sample_given_totals(resp.to_numpy(dtype=float), thr, rng)
            clone.responses = pd.DataFrame(sim, index=resp.index, columns=resp.columns)
            try:
                fits = [
                    clone._global_fit_subset(groups[n], items=items, cml_hessian=hess[n])
                    for n in names
                ]
                f0 = clone._global_fit_subset(both, items=items, cml_hessian=h0)
            except ValueError:
                skipped += 1
                continue
            if any(len(m.item_names) < len(items) for m in fits + [f0]):
                skipped += 1  # an item without CML estimates: not comparable
                continue
            lr = 2.0 * (
                sum(m.convergence["log_likelihood"] for m in fits)
                - f0.convergence["log_likelihood"]
            )
            exceed += lr >= LR - 1e-9 * max(1.0, abs(LR))
            used += 1
            if mc_exceedances is not None and exceed >= mc_exceedances:
                stopped = True  # Besag-Clifford sequential stopping
                break
        if not used:
            raise ValueError(
                "method='andersen': no Monte Carlo sample had CML estimates for every item; "
                "try unused_categories='null'."
            )
        p = exceed / used if stopped else (1 + exceed) / (1 + used)
        return p, used, skipped, stopped

    def _composite_layout(self, items):
        """global_fit(method='andersen') under 'cp-full-rcl': (maximum scores, free directions in the stacked
        uncentred thresholds, a function stacking a fit's uncentred thresholds) for these items."""
        from raschpy import _rcl, _selection
        from raschpy.pcm import PCM
        from raschpy.rsm import RSM

        I = len(items)
        if isinstance(self, RSM):
            K = int(self.max_score)
            return [K] * I, _selection.bases(I, K)[0], lambda m: self._stacked(rsm=m, names=items, K=K)
        if isinstance(self, PCM):
            Ks = [int(k) for k in self.max_score_vector.loc[items]]
            return Ks, _rcl.zero_sum_basis(sum(Ks)), lambda m: self._stacked(pcm=m, names=items)
        return [1] * I, _rcl.zero_sum_basis(I), lambda m: m.items.reindex(items).to_numpy(float)

    def _composite_fits(self, src, groups, names, both, items):
        """The pooled and group 'cp-full-rcl' fits of src's responses: the pooled fit with the object's
        calibrate() settings, and with PCM's ridge each group at the pooled fit's ridge weight. Returns
        (pooled, {group: fit})."""
        ridged = hasattr(src, "_fixed_ridge")
        if ridged:
            src.ridge = None  # the pooled fit applies the object's ridge setting itself
        pooled = src._global_fit_subset(both, items=items)
        if ridged:
            src.ridge = getattr(pooled, "ridge", None)
        return pooled, {n: src._global_fit_subset(groups[n], items=items) for n in names}

    def _composite_andersen(
        self, split, covariate, seed, split_prop, alpha, p_method, no_of_samples, mc_exceedances
    ):
        """
        global_fit(method='andersen') for 'cp-full-rcl' estimates: Andersen's test on the pairwise
        conditional (composite) likelihood (_selection.composite_andersen). p_method='monte-carlo': the
        adjusted composite LR against datasets simulated from the pooled fit given each person's total and
        items, every fit repeated (Besag-Clifford stopping with mc_exceedances); 'chi2': the composite LR
        with the exact weighted chi-square tail. Items are those responded to in every group and kept by every fit.
        """
        from raschpy import _esf, _selection

        groups = self._global_fit_groups(split, covariate, seed, split_prop, max_groups=np.inf)
        names = list(groups)
        both = groups[names[0]].append([groups[n] for n in names[1:]])
        work = copy.copy(self)
        common = [
            c
            for c in self.item_names
            if all(self.responses.loc[groups[n], c].notna().any() for n in names)
        ]
        while len(common) >= 2:
            pooled, fitted = self._composite_fits(work, groups, names, both, common)
            kept = [c for c in common if all(c in m.item_names for m in [pooled, *fitted.values()])]
            if kept == common:
                break
            common = kept
        if len(common) < 2:
            raise ValueError(
                "method='andersen': fewer than two items have estimates in every group, so there is "
                "nothing to compare."
            )
        Ks, Bm, stack = self._composite_layout(common)

        def test(src, pooled, fitted):
            Xs = [src.responses.loc[groups[n], common].to_numpy(float) for n in names]
            return _selection.composite_andersen(
                Xs, Ks, [stack(fitted[n]) for n in names], stack(pooled), Bm
            )

        res = test(work, pooled, fitted)
        b0 = stack(pooled)
        off = np.concatenate([[0], np.cumsum(Ks)]).astype(int)
        thr = [b0[off[i] : off[i + 1]] for i in range(len(common))]
        p, used, skipped, stopped = res["p"], None, None, None
        if p_method == "monte-carlo":
            rng = np.random.default_rng(seed)
            resp = self.responses.loc[both, common]
            X0 = resp.to_numpy(dtype=float)
            T = res["adjusted"]
            exceed, used, skipped, stopped = 0, 0, 0, False
            for _ in _progress.track(range(no_of_samples), "Andersen LR Monte Carlo p-value", unit="samples",
                                     alternative="mc_exceedances= stops early once the p-value is clearly "
                                     "large; p_method='chi2' takes seconds."):
                work.responses = pd.DataFrame(
                    _esf.sample_given_totals(X0, thr, rng), index=resp.index, columns=common
                )
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        f0, fg = self._composite_fits(work, groups, names, both, common)
                        if any(len(m.item_names) < len(common) for m in [f0, *fg.values()]):
                            skipped += 1  # an item lost in a fit: not comparable
                            continue
                        ts = test(work, f0, fg)["adjusted"]
                except (ValueError, np.linalg.LinAlgError):
                    skipped += 1
                    continue
                exceed += ts >= T - 1e-9 * max(1.0, abs(T))
                used += 1
                if mc_exceedances is not None and exceed >= mc_exceedances:
                    stopped = True  # Besag-Clifford sequential stopping
                    break
            if not used:
                raise ValueError(
                    "method='andersen': no Monte Carlo sample kept every item in every fit; use "
                    "p_method='chi2'."
                )
            p = exceed / used if stopped else (1 + exceed) / (1 + used)
        self._andersen_item_tables(fitted, pooled, groups, names, thr=thr, item_table=False)
        excluded = [c for c in self.item_names if c not in common]
        self.global_fit_table = None
        self.global_fit_excluded = excluded
        self.global_fit_groups = fitted
        statistic = res["adjusted"] if p_method == "monte-carlo" else res["lr"]
        summary = {"method": "andersen", "estimation": "cp-full-rcl", "split": split,
                   "statistic": statistic, "df": res["df"]}
        if split in ("score", "person_location"):
            summary["split_prop"] = split_prop
        if split == "exogenous":
            summary["covariate"] = covariate
        summary.update(
            {
                "composite LR": res["lr"],
                "adjusted composite LR": res["adjusted"],
                "Satterthwaite scale": res["scale"],
                "Satterthwaite df": res["df_satterthwaite"],
                "composite log-likelihoods": {
                    **dict(zip(names, res["cl"])), "pooled": float(sum(res["cl_pooled"]))
                },
                "p_method": p_method,
                "p-value": p,
                "p-value (chi-square)": res["p"],
                "Monte Carlo samples": used,
                "Monte Carlo samples skipped": skipped,
                "Monte Carlo stopped early": stopped,
                "reject": bool(p <= alpha),
                "group sizes": {n: len(groups[n]) for n in names},
                "items excluded": len(excluded),
            }
        )
        self.global_fit_summary = pd.Series(summary, name="Global fit")
        from raschpy.pcm import PCM

        if p_method == "chi2" and isinstance(self, PCM):
            warnings.warn(
                "The chi-square p-value of Andersen's test under 'cp-full-rcl' is liberal for PCM; "
                "p_method='monte-carlo' is recommended.",
                UserWarning,
                stacklevel=3,
            )
        if excluded:
            warnings.warn(
                f"{len(excluded)} item(s) left out of the likelihood ratio test because they have no "
                f"estimates in one group: {excluded[:10]}{' ...' if len(excluded) > 10 else ''}. See "
                f"global_fit_excluded.",
                UserWarning,
                stacklevel=3,
            )
        return self.global_fit_summary

    def _obsexp_variances(self, pooled, X, thr, labels, G, items):
        """
        Variance of each item's observed - expected score total in each group (labels 0..G-1; -1 = left
        out) under the pooled fit, (n_items, G), allowing for the estimation of its item parameters
        (_threshold_influence, _corrected_group_variances).
        """
        Phi, A = pooled._threshold_influence(X, items)
        return self._corrected_group_variances(X, thr, labels, G, A, Phi)

    @staticmethod
    def _corrected_group_variances(X, thr, labels, G, A, Phi, sets=None):
        """
        Variance of each item's observed - expected score total in each group (labels 0..G-1; -1 = left
        out), (n_items, G), at fitted thresholds thr, allowing for their estimation; with sets (lists of
        item positions), of each set's total instead, (len(sets), G). Phi (persons x stacked
        thresholds) is each person's contribution to the estimates' deviation from their limit (the
        estimator's influence function), so that each group's threshold-level deviations are, to first
        order, d_g + V_g sum_n Phi_n (d_n a person's indicators x >= k minus their conditional
        expectations given the person's total, d_g their sum over the group, V_g the group's conditional
        covariance of the indicators). Writing Phi_n = -B d_n + w_n, with B = A (A'VA)^-1 A' the influence
        of CML (A the free directions, V the sum over all persons), the d_n parts use the model's
        covariance and only w_n, the estimator's departure from CML, is estimated from the persons:
        W_g = V_g - V_g B V_g + C_g + C_g' + V_g J V_g, C_g = (F_g - V_g B F) V_g, F_g = sum over the group
        of d_n w_n', F over all persons, J the sum of w_n w_n' (w_n centred). For unpenalised CML w_n = 0,
        and this is Glas's (1988) within-group covariance V_g - V_g B V_g.
        """
        from raschpy import _esf

        Ks = [len(t) for t in thr]
        off = np.concatenate([[0], np.cumsum(Ks)]).astype(int)
        if sets is None:
            sets = [[i] for i in range(len(Ks))]
        cols = [np.concatenate([np.arange(off[i], off[i + 1]) for i in st]).astype(int) for st in sets]
        S = _esf.person_scores(X, thr)
        d = -S
        Phi = np.asarray(Phi, float)
        est = (np.abs(S).sum(1) > 0) | (np.abs(Phi).sum(1) > 0)
        V = -_esf.cond_loglik(X[est], thr, hessian=True)[2]
        H = A.T @ V @ A
        B = A @ np.linalg.pinv((H + H.T) / 2, rcond=1e-10, hermitian=True) @ A.T
        w = np.where(est[:, None], Phi + d @ B, 0.0)
        w = np.where(est[:, None], w - w[est].mean(0), 0.0)
        J, F = w.T @ w, d.T @ w
        out = np.full((len(sets), G), np.nan)
        for g in range(G):
            sel = labels == g
            Vg = -_esf.cond_loglik(X[sel], thr, hessian=True)[2]
            VgB = Vg @ B
            C = (d[sel].T @ w[sel] - VgB @ F) @ Vg
            Wg = Vg - VgB @ Vg + C + C.T + Vg @ J @ Vg
            for j, c in enumerate(cols):
                out[j, g] = Wg[np.ix_(c, c)].sum()
        return out

    def _threshold_influence(self, X, items):
        """
        Each person's contribution to the deviation of the fitted uncentred thresholds (stacked item by
        item, as _cml_threshold_list) from their limit, to first order, for responses X to items (the
        object's responses and items), with the estimator's free directions A: (Phi, A). CML: the
        conditional scores times the inverse of the (penalised) information; the other estimators from
        _estimator_influence(). Raises ValueError for other items.
        """
        from raschpy import _esf

        if list(items) != list(self.item_names):
            raise ValueError("The influence functions are for the fit's items.")
        thr = self._cml_threshold_list(items)
        m = sum(len(t) for t in thr)
        if getattr(self, "estimation", None) == "cml":
            A = self._cml_design(m)
            S = _esf.person_scores(X, thr)
            keep = np.abs(S).sum(1) > 0
            H = A.T @ (-_esf.cond_loglik(X[keep], thr, hessian=True)[2]) @ A
            Q = getattr(self, "_cml_penalty", None)
            if Q is not None:
                H = H + Q
            Hinv = np.linalg.pinv((H + H.T) / 2, rcond=1e-10, hermitian=True)
            return S @ A @ Hinv @ A.T, A
        return self._estimator_influence(X), self._composite_layout(items)[1]

    @staticmethod
    def _priority_vector_derivative(P, theta, method, dP):
        """
        Derivatives of priority_vector(P, method) (zero-centred measures theta) in the directions dP of
        the matrix, (n, m, m): (n, m). 'log-lik' by implicit differentiation of the Bradley-Terry
        equations it solves; 'ls', 'cos' and 'evm' (the first principal axis of the column-centred
        reciprocal matrix) directly.
        """
        m = P.shape[0]
        off = ~np.eye(m, dtype=bool)
        if method == "log-lik":
            # strengths beta = -theta; pi[i, j] = sigma(beta_i - beta_j)
            with np.errstate(over="ignore"):
                pi = off / (1.0 + np.exp(-(theta[None, :] - theta[:, None])))
            W = (P + P.T) * pi * (1.0 - pi) * off
            Lp = Rasch._origin_pinv(np.diag(W.sum(1)) - W)
            dg = (dP * ((1.0 - pi) * off)[None]).sum(2) - (dP.transpose(0, 2, 1) * pi[None]).sum(2)
            out = -(dg @ Lp)
        else:
            valid = off & (P > 0)
            with np.errstate(divide="ignore", invalid="ignore"):
                R = np.where(valid, P.T / P, 1.0)
                dR = np.where(valid[None], (dP.transpose(0, 2, 1) - R[None] * dP) / np.where(valid, P, 1.0)[None], 0.0)
            if method == "ls":
                out = dR.mean(2) / R.mean(1)[None]
            elif method == "evm":
                Xc = R - R.mean(0)
                vals, vecs = np.linalg.eigh(Xc.T @ Xc)
                v = vecs[:, -1]
                Mp = (vecs[:, :-1] / (vals[-1] - vals[:-1])) @ vecs[:, :-1].T
                dXc = dR - dR.mean(1, keepdims=True)
                dCv = np.einsum("nij,i->nj", dXc, Xc @ v) + (dXc @ v) @ Xc
                out = -(dCv @ Mp) / v[None]
            else:  # 'cos'
                nrm = np.linalg.norm(R, axis=0)
                w = (R / nrm[None]).sum(1)
                dn = (R[None] * dR).sum(1) / nrm[None]
                dw = (dR / nrm[None, None]).sum(2) - (R[None] * (dn / nrm**2)[:, None, :]).sum(2)
                out = dw / w[None]
        return out - out.mean(1, keepdims=True)

    def _pair_influence(self, X):
        """
        Each person's contribution to the deviation of PAIR's measures (priority_vector() of the
        smoothed, possibly powered pairwise matrix, as calibrate() computed them) from their limit, to
        first order, for the object's responses X: the derivative of the measures in the direction of the
        person's comparisons (_pair_terms: the person's part of the raw matrix is the sum of the outer
        products u v' of its terms, times the mask; the smoothing constants belong to no one). (persons, m).
        """
        st = self._pair_state
        M, P, k = st["M"], st["P"], int(st["power"])
        terms, mask = self._pair_terms(X)
        powers = [np.linalg.matrix_power(M, j) for j in range(k)]
        N, m = X.shape[0], M.shape[0]
        out = np.zeros((N, m))
        chunk = max(1, int(4e6 // (m * m)))
        for s0 in range(0, N, chunk):
            sl = slice(s0, min(N, s0 + chunk))
            n = sl.stop - sl.start
            if k == 0 or mask is not None:
                dM = np.zeros((n, m, m))
                for U, V in terms:
                    dM += U[sl][:, :, None] * V[sl][:, None, :]
                if mask is not None:
                    dM *= mask[None]
                # d(M^k) = sum_j M^j dM M^(k-1-j)
                dP = dM if k == 0 else sum(powers[j] @ dM @ powers[k - 1 - j] for j in range(k))
            else:
                # the same for a person's dM = sum of u v', one vector product at a time
                dP = np.zeros((n, m, m))
                for U, V in terms:
                    u, v = U[sl], V[sl]
                    for j in range(k):
                        dP += (u @ powers[j].T)[:, :, None] * (v @ powers[k - 1 - j])[:, None, :]
            out[sl] = self._priority_vector_derivative(P, st["theta"], st["method"], dP)
        return out

    def _andersen_item_tables(self, fitted, pooled, groups, names, thr=None, item_table=True):
        """
        global_fit(method='andersen') follow-up, showing which items differ between the groups:
        global_fit_obsexp (per item and group: persons, observed and expected mean item score under the
        pooled fit, the expectation conditional on each person's total and items, and the standardised
        difference, its variance allowing for the estimation of the item parameters at the pooled fit
        (_obsexp_variances)) and global_fit_item_table (per item: each group's CML item location,
        their difference and Wald z, and the conditional item chi-square over the groups with df =
        groups - 1, its p-value and BH-adjusted p-value). thr: the pooled fit's uncentred thresholds per item
        (default its CML thresholds); item_table=False sets global_fit_item_table to None ('cp-full-rcl').
        """
        from raschpy import _esf

        items = list(pooled.item_names)
        X = np.array(pooled.responses.loc[:, items], dtype=np.float64)
        mom = _esf.conditional_moments(X, pooled._cml_threshold_list(items) if thr is None else thr)
        labels = np.full(len(X), -1)
        for k, n in enumerate(names):
            labels[pooled.responses.index.isin(groups[n])] = k
        st = self._q1_statistics(X, mom["E"], mom["V"], labels)
        var = self._obsexp_variances(pooled, X, pooled._cml_threshold_list(items) if thr is None else thr,
                                     labels, len(names), items)
        rows = []
        for i, item in enumerate(items):
            for k, n in enumerate(names):
                cnt, v = st["n"][i, k], var[i, k]
                rows.append(
                    {
                        "Persons": int(cnt),
                        "Observed mean": st["O"][i, k] / cnt if cnt else np.nan,
                        "Expected mean": st["E"][i, k] / cnt if cnt else np.nan,
                        "z": (
                            (st["O"][i, k] - st["E"][i, k]) / np.sqrt(v)
                            if st["V"][i, k] > 0 and v > 1e-10
                            else np.nan
                        ),
                    }
                )
        self.global_fit_obsexp = pd.DataFrame(
            rows,
            index=pd.MultiIndex.from_product([items, names], names=["Item", "Group"]),
        )
        if not item_table:
            self.global_fit_item_table = None
            return
        tab = pd.DataFrame(index=items)
        se = {}
        for n in names:
            m = fitted[n]
            if getattr(m, "_cml_marked", None) is not None:  # unused_categories='null'
                item_se = m._cml_location_se()
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    m.std_errors()
                item_se = m.item_se
            tab[f"Location ({n})"] = m.items.reindex(items).values
            se[n] = item_se.reindex(items).values
        if len(names) == 2:
            a, b = names
            tab["Difference"] = tab[f"Location ({b})"] - tab[f"Location ({a})"]
            with np.errstate(invalid="ignore", divide="ignore"):
                tab["z"] = tab["Difference"] / np.sqrt(se[a] ** 2 + se[b] ** 2)
        df = st["df"].astype(float)
        tab["Chi-square"] = st["q1"]
        tab["df"] = st["df"]
        tab["p"] = np.where(df > 0, chi2_dist.sf(st["q1"], np.maximum(df, 1)), np.nan)
        tab["p (BH)"] = self._bh_correction(tab["p"].fillna(1.0)).where(
            tab["p"].notna()
        )
        self.global_fit_item_table = tab

    def _global_fit_needs_bootstrap(self):
        """True if the object's calibration settings have no sandwich
        covariance (method='cos', 'ls' or 'evm', for SLM, PCM, and RSM with
        estimation='pair-cpat'), so global_fit() uses bootstrap covariances."""
        if getattr(self, "estimation", "pair-cpat") not in ("pair", "pair-cpat"):
            return False
        return (self._calibration_kw().get("method") or "log-lik") != "log-lik"

    def _global_fit_bootstrap_cov(self, idx, labels, no_of_samples, rng):
        """Bootstrap covariance of a global_fit() group's estimates (as
        returned by _global_fit_params) over no_of_samples resamples of its
        persons; resamples whose parameters differ (an item lost) are
        skipped."""
        rows = []
        for _ in _progress.track(range(no_of_samples), "Global fit bootstrap", unit="resamples"):
            draw = idx[rng.integers(0, len(idx), len(idx))]
            m = self._global_fit_subset(draw)
            lab, est, _, _, _ = self._global_fit_params(m, covariance=False)
            if list(lab) == list(labels):
                rows.append(est)
        return np.cov(np.array(rows), rowvar=False)

    @staticmethod
    def _global_fit_align(parts):
        """
        global_fit()'s two groups' (labels, estimates, covariance, comparable
        mask, location blocks) re-indexed onto the union of their labels,
        for groups that responded to different items: a parameter missing from a
        group is marked not comparable. Returns (parts, blocks), blocks as
        positions in the union.
        """
        labels = list(parts[0][0]) + [x for x in parts[1][0] if x not in parts[0][0]]
        pos = {x: i for i, x in enumerate(labels)}
        n = len(labels)
        out = []
        for lab, est, V, ok, _ in parts:
            ix = np.array([pos[x] for x in lab])
            e2 = np.zeros(n)
            V2 = np.zeros((n, n))
            ok2 = np.zeros(n, bool)
            e2[ix] = est
            V2[np.ix_(ix, ix)] = V
            ok2[ix] = ok
            out.append((labels, e2, V2, ok2, None))
        blocks = [
            np.array(sorted({pos[parts[g][0][j]] for g in (0, 1) for j in blk[g]}))
            for blk in zip(parts[0][4], parts[1][4])
        ]
        return out, blocks

    @_progress.with_progress
    def global_fit(
        self,
        method="wald",
        split=None,
        covariate=None,
        welch=None,
        alpha=0.05,
        seed=None,
        split_prop=0.5,
        se_correction="kauermann-carroll",
        no_of_samples=500,
        p_method=None,
        mc_exceedances=None,
        unused_categories=None,
    ):
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
        and reported. W is referred to the multivariate Welch F reference of
        Nel and van der Merwe (1986): F = W (nu - df + 1) / (df nu) on
        (df, nu - df + 1), with nu = [tr(V^2) + tr(V)^2] / sum over the groups
        of [tr(V_g^2) + tr(V_g)^2] / (n_g - 1), V = V_1 + V_2 and n_g the
        group sizes. It allows for estimating each group's covariance from
        its own persons, so the smaller group counts for more when the groups
        differ in size; a chi-square reference (the large-sample limit)
        ignores that and is liberal with many parameters per person, and the
        pooled two-sample Hotelling reference (df from n_1 + n_2) is liberal
        when the groups differ in size. With one parameter, nu is the
        Welch-Satterthwaite df of dif_test. The F reference is an
        approximation, not an exact F test: it is exact only when the
        persons' contributions to d are multivariate normal and the
        covariance is their sample covariance, whereas d here is a difference
        of estimates and V a sandwich (or bootstrap) covariance. The sandwich
        covariances use the Kauermann-Carroll small-sample correction by
        default (se_correction): with few persons per parameter the
        uncorrected sandwich is too small and the test liberal. In null
        simulations (SLM, RSM and PCM; 12 items, maximum score 3 for RSM and
        PCM; N = 200, 600 and 2,000; random and score splits, and an
        exogenous split with groups of 30% and 70% of the persons; 500
        datasets per design, 1,000 for PCM), the rejection rate at alpha =
        0.05 was 2.4-7.3% with the correction (0.2-2.0% at alpha = 0.01) and
        2.8-9.7% without it. The highest was PCM at N = 200 on the random
        split (7.3%: about 100 persons per group for 35 parameters, where the
        spread of W, not the reference's degrees of freedom, runs ahead of
        the F reference). On the exogenous split, PCM at N = 200 (groups of
        about 60 and 140) rejected 4.6%, against 16.9% with the pooled
        two-sample Hotelling reference (df from n_1 + n_2) and 47.5% with
        chi-square. Random splits have essentially no power against misfit
        (both halves contain
        the same mix of persons), so they serve as a calibration check; power
        comes from splits correlated with the misfit -- the score split for
        discrimination differences, an exogenous variable for DIF.

        method='andersen': Andersen's conditional likelihood ratio test
        (SLM, PCM, RSM; CML estimates, or 'cp-full-rcl' estimates as below),
        LR = 2 (sum of the groups' ll - ll_pooled), each the conditional
        log-likelihood at its CML maximum, referred to chi-square with
        df = (groups - 1) x the model's free parameters. The pooled model is
        fitted to the persons of the groups. Items without CML estimates in
        every group (an item one group
        did not respond to, an unused category, or an item not compared with
        the others in both directions) are left out of every fit, with a
        UserWarning; with PCM, the score and person-location splits often
        leave out items whose top category is unused in the low group or
        bottom category in the high group. In null simulations (SLM, PCM and
        RSM; I = 10, K = 3 for PCM/RSM; N = 500 and 2,000; score, random,
        exogenous and person-location splits; 300 datasets each), the
        rejection rate at alpha = 0.05 was 0.023-0.070 and the mean LR/df
        0.94-1.04; PCM's score split left out 3.8 of 10 items on average at
        N = 500 and 1.9 at N = 2,000. With PAIR or 'cpat-rcl' estimates,
        which do not maximise the likelihood the statistic evaluates, it
        raises a ValueError: in simulation it then floored at 0 on score and
        person-location splits, and its null distribution was not a rescaled
        chi-square. To show which items differ, it also sets
        global_fit_obsexp (per item and group: persons, observed and
        expected mean item score under the pooled fit, the expectation
        conditional on each person's total and items, and the standardised
        difference, z, whose variance allows for the estimation of the item
        parameters at the pooled fit, Glas's correction) and global_fit_item_table (per item: each group's
        CML location, their difference and Wald z, and the item's
        conditional chi-square over the groups, df = groups - 1, with p and
        BH-adjusted p).

        With estimates from calibrate(estimation='cp-full-rcl') (the default),
        the test is on the pairwise conditional likelihood those estimates
        maximise: the groups and their pooled persons are refitted with
        'cp-full-rcl' (PCM's groups at the pooled fit's ridge weight), and the
        composite LR is 2 (sum of the groups' composite log-likelihoods at their
        fits - at the pooled fit), unpenalised. Persons appear in many item
        pairs, so its null distribution is not chi-square. p_method='monte-carlo'
        (the default here) refers the adjusted composite LR (Pace, Salvan and
        Sartori, 2011: the LR multiplied by the ratio of the sandwich Wald
        statistic of the group differences to the Wald statistic from the
        likelihood's curvature) to datasets simulated from the pooled fit
        given each person's total and items, every fit repeated. 'chi2' refers
        the composite LR to a weighted sum of chi-square(1) variables (Kent,
        1982), with that sum's exact tail probability. In null simulations (SLM, PCM
        and RSM with maximum score 3 or 4; 10 or 20 items; 200, 500 and 2,000
        persons; complete and booklet designs; score, score-group, random and
        exogenous splits; 1,200 datasets per model and maximum score, each
        split four ways) the Monte Carlo test
        rejected 0.043-0.054 of datasets at alpha = 0.05 by model (CML's test
        0.044-0.056), and 0.050 with three exogenous groups. With sparse top
        categories (PCM, maximum score 4, persons centred 1.5 logits below the
        items) it rejected 0.020-0.040. The chi-square p-value (the designs
        above with 50 datasets per design, 13,200 tests) rejected 0.055 for
        SLM, 0.054-0.058 for RSM and 0.047-0.059 for PCM by maximum score (by
        split 0.033-0.067), and with sparse top categories 0.070 for PCM and
        0.038 for RSM, so for PCM a UserWarning recommends
        p_method='monte-carlo'. The Monte Carlo test had
        less power than CML's with 200 and 500 persons (0.5 logits of DIF on two
        items, exogenous split: PCM 0.10-0.17 lower, RSM 0.02-0.14, SLM
        0.04-0.06) and much the same with 2,000; for the most power, calibrate with
        estimation='cml'. global_fit_obsexp is set as for CML (expectations
        under the pooled fit's thresholds; z's variance allows for the
        'cp-full-rcl' estimation at that fit, from each person's contribution
        to its estimating equations); global_fit_item_table is None.

        method='martin-lof': Martin-Lof's test of unidimensionality for two
        item subsets (split=), SLM, PCM and RSM, estimates from
        calibrate(estimation='cml') only. LR = 2 (ll_A + ll_B - ll_all +
        sum n_rs log(n_rs / N) - sum n_t log(n_t / N)), the conditional
        log-likelihoods of the two subsets' and all items' CML fits plus the
        persons' subtotal (r on A, s on B) and total (t) distributions, both
        within each missingness pattern; df = sum of M_A M_B over patterns +
        q_A + q_B - q (M: maximum subtotal; q: the fits' free parameters). For
        RSM, each subset has its own category widths, so df is K - 1 larger
        than for PCM with the same maximum scores. The chi-square reference
        is conservative when the subtotal table is sparse: in null
        simulations (SLM, PCM, RSM; I = 6 and 10; K = 2 and 3; N = 500 to
        10,000; complete and booklet designs) it held its level (rejection
        rate 0.027-0.067 at alpha = 0.05) only when at most about a quarter
        of the table's cells had fewer than 5 persons, which few designs
        reach, and fell to 0 as that share grew. A UserWarning is then raised,
        naming p_method='monte-carlo': the p-value from datasets simulated
        from the joint fit given each person's total and observed items, each
        refitted (in timing tests, 0.03-13 s per sample, from 10 items and
        500 persons to 30 items, 2,000 persons and 20% missing responses).
        In the sparsest of those designs (PCM and RSM, I = 10, K = 3; 150-300
        datasets each, 200 samples) its rejection rate was 0.027-0.077 (the highest, RSM
        with N = 2,000, over 300 datasets), against 0.000-0.013 for the
        chi-square p-value.

        Parameters
        ----------
        method : {'wald', 'andersen', 'martin-lof'}, default 'wald'
            'andersen' needs estimates from calibrate(estimation='cml') or
            estimation='cp-full-rcl'; 'martin-lof' from estimation='cml'.
        split : str, list or None, default None
            'wald' and 'andersen': how persons are split, one of 'score'
            (the default), 'score_groups', 'person_location', 'random' or
            'exogenous': at the median mean observed score (the mean handles
            missing data; persons with the same score may fall in different
            groups, so the groups are of equal size); into two groups of whole
            score groups, cut where the persons with non-extreme scores divide
            most evenly (as iarm's clr_tests does; split_prop is not used); at
            the median person location; at random; or by an exogenous
            covariate, two-valued for 'wald', any number of values for
            'andersen'. 'martin-lof': how items are split,
            'median' (the default: the items below and above the median item
            location) or a list of the items in the first subset.
        covariate : str or None, default None
            Column of self.exogenous, for split='exogenous' (implied when split
            is left out; any other split raises a ValueError).
        split_prop : float, default 0.5
            For split='score' or 'person_location': the proportion of persons
            in each of the two outer groups compared. 0.5 is the median split
            (everyone); 1/3 compares the lowest and highest thirds, leaving out
            the middle third; 0.4 the lowest and highest 40%. Smaller values
            give groups further apart in ability (a stronger contrast for
            discrimination differences) but fewer persons and more extreme
            targeting in each group.
        welch : bool or None, default None
            Deprecated (FutureWarning; to be removed in a future version).
            welch=True gives the previous reference, W/df on F(df, n_s - df)
            with n_s the smaller group's size; None or False gives the
            multivariate Welch F reference above, which is more accurate.
        alpha : float, default 0.05
            Significance level for the decision.
        seed : int or None, default None
            Seed for split='random'.
        se_correction : {'kauermann-carroll', None}, default 'kauermann-carroll'
            Small-sample correction of each group's sandwich covariance (Wald
            only; CML's too, unless se_method='hessian'): each person's score
            contribution u_n is replaced by (I - H_n)^(-1/2) u_n, H_n being the
            person's leverage (Kauermann & Carroll, 2001; Bell & McCaffrey,
            2002), as in std_errors(). Negligible with many persons per
            parameter. None uses the uncorrected sandwich.
        no_of_samples : int, default 500
            Wald: bootstrap resamples per group, used instead of the sandwich
            when the object was calibrated with method='cos', 'ls' or 'evm'
            (SLM, PCM; RSM with estimation='pair-cpat'): the sandwich is the
            covariance of method='log-lik' only. se_correction does not
            apply then. 'martin-lof' and 'andersen' with
            p_method='monte-carlo': the number of simulated datasets.
        p_method : {None, 'chi2', 'monte-carlo'}, default None
            'martin-lof' and 'andersen': the p-value from the chi-square
            reference, or from no_of_samples datasets simulated from the joint
            (Andersen: pooled) CML fit given each person's total and observed
            items (each refitted and its statistic recomputed, Andersen's with
            the observed groups; seed= for reproducibility; 'cp-full-rcl': see
            method='andersen' above). Each refit
            starts from the PAIR (RSM: cp-full-rcl) estimates and uses the
            observed fit's Hessian as a fixed Newton matrix (the chord
            method), converging to the same maximum with gradient evaluations
            only. Simulated datasets in which an item has no CML estimates are
            left out and counted. None: 'monte-carlo' for 'andersen' with
            'cp-full-rcl' estimates and with PCM's CML ridge (calibrate(estimation='cml'), ridge > 0), and
            for 'andersen' with RSM's CML when a group's or the pooled fit
            needed the unused-category width ridge (a score category unused
            in that fit; with it the chi-square p-value was conservative in
            null simulations, rejecting 3% at alpha = 0.05); else 'chi2'. With the ridge each group's fit is pulled towards the
            common threshold pattern more than the pooled fit (fewer persons,
            same weight), which shrinks the likelihood ratio, so the
            chi-square p-value is conservative: in null simulations (PCM, 10
            items, maximum score 3, 5 or 8, N = 100 to 500, 100 datasets per
            design) it rejected 0% of the time at alpha = 0.05 with the
            default split ('score') and 2% with a random split; Monte Carlo
            rejected 3.6% and 3.7%. When None chooses 'monte-carlo',
            mc_exceedances=None means 10.
        mc_exceedances : int or None, default None
            'martin-lof' or 'andersen' with p_method='monte-carlo': stop sampling once this
            many simulated statistics reach the observed one (Besag and
            Clifford's sequential p-value, h / samples drawn), so a clearly
            non-significant result needs only a few dozen samples; None draws
            all no_of_samples. E.g. 10.
        unused_categories : {None, 'drop', 'null'}, default None
            'andersen' with CML estimates (PCM and RSM; None follows the object's
            calibrate(unused_categories=...)): a score category that no person with a
            non-extreme total used, in one group or the pooled data. 'drop'
            leaves out an item with such a category (PCM; for RSM, a category
            unused by every item raises a ValueError). 'null' keeps the item
            with that category's probability zero, the scores unchanged
            (Wilson and Masters, 1993; psychotools' pcmodel(nullcats='keep')):
            the data identify only the sum of the two thresholds either side
            of it, so that fit has one parameter fewer for each such category,
            and df = (the groups' free parameters) - (the pooled fit's).
            The chi-square p-value is then liberal with sparse categories: in
            null simulations (PCM, 10 items scored 0-4, persons centred 1.5
            logits below the items, N = 300 and 1,000, score-group and random
            splits, 300 datasets each) it rejected 7-25% of the time at
            alpha = 0.05 (with a UserWarning); p_method='monte-carlo' (199
            samples) rejected 3.7-4.3% in three of the four designs and 8.7% in
            the fourth (N = 300, random split). For RSM on the same design both
            p-values held 5.0-7.0%. On iarm's DESC-II data the log-likelihoods
            and parameter counts match psychotools' pcmodel and rsmodel exactly.

        Each group is calibrated with the object's last calibrate()
        settings. In null simulations under method='cos', 'ls' or 'evm'
        (bootstrap covariances), PCM's test was liberal (0.08-0.10 at
        alpha = 0.05, 300 persons per group; SLM and RSM 0.03-0.065). For a
        test with a nominal false-positive rate,
        re-run with method='log-lik': pcm.calibrate(method='log-lik'), then
        pcm.global_fit(...) (0.05-0.07), or use the per-item tests of
        pcm.dif_test() (0.045-0.062 with bootstrap standard errors).

        Attributes set
        --------------
        global_fit_summary : pandas.Series
            Method, split, statistic, df, (denominator df), covariance
            ('sandwich', 'bootstrap' or 'hessian'), se_correction, p-value,
            decision; for 'andersen', the conditional log-likelihoods of
            the groups and the pooled model instead of the covariance, and
            with 'cp-full-rcl' estimates the statistic the p-value refers to
            (the adjusted composite LR with 'monte-carlo', the composite LR
            with 'chi2'), both statistics, the Satterthwaite scale and df,
            the composite log-likelihoods, and the chi-square p-value.
        global_fit_table : pandas.DataFrame or None
            Wald only: each compared parameter's difference, SE and z.
        global_fit_excluded : list
            Parameters (Wald) or items ('andersen') not compared, not
            estimable in both groups.
        global_fit_groups : dict
            {group name: calibrated model}.

        Each result is also kept in global_fit_results, keyed by method, split and its options, so
        that global_stats_df() can report every test run, not only the latest.
        """
        if p_method is None:
            ridged = getattr(self, "estimation", None) == "cml" and (getattr(self, "ridge", None) or 0) > 0
            if method == "andersen" and getattr(self, "estimation", None) == "cp-full-rcl":
                p_method = "monte-carlo"  # the adjusted composite LR
            elif method == "andersen" and ridged:
                p_method = "monte-carlo"
            elif method == "andersen" and hasattr(self, "_width_ridge"):
                p_method = "auto"  # RSM: Monte Carlo if a fit used the unused-category width ridge (_andersen_lr)
            else:
                p_method = "chi2"
            if p_method == "monte-carlo" and mc_exceedances is None:
                mc_exceedances = 10  # chosen by default: stop early when clearly not significant
        if welch is not None:
            warnings.warn(
                "global_fit(welch=...) is deprecated and will be removed in a future version. The Wald test "
                "now refers its statistic to the multivariate Welch F reference (Nel and van der Merwe, 1986), "
                "with denominator degrees of freedom from each group's covariance and size, which is more "
                "accurate than the previous reference (welch=True: degrees of freedom from the smaller group "
                "only) and than chi-square (which ignores that the covariances are estimated). "
                + ("welch=True still gives the previous reference until then." if welch else
                   "welch=False no longer gives chi-square; the F reference is used.")
                + " Drop the argument to use the F reference.",
                FutureWarning,
                stacklevel=2,
            )
        out = self._global_fit_run(
            method=method,
            split=split,
            covariate=covariate,
            welch=bool(welch),
            alpha=alpha,
            seed=seed,
            split_prop=split_prop,
            se_correction=se_correction,
            no_of_samples=no_of_samples,
            p_method=p_method,
            mc_exceedances=mc_exceedances,
            unused_categories=unused_categories,
        )
        if split is None:
            split = (
                "median"
                if method == "martin-lof"
                else ("exogenous" if covariate is not None else "score")
            )
        split_label = (
            "items " + ", ".join(map(str, split))
            if isinstance(split, (list, tuple))
            else split
        )
        details = [f"split={split_label}"]
        if covariate is not None:
            details.append(f"covariate={covariate}")
        if method == "martin-lof" or (method == "andersen" and p_method != "chi2"):
            details.append(f"p_method={p_method}")
        if method == "andersen" and unused_categories == "null":
            details.append("unused_categories=null")
        if method == "wald" and welch:
            details.append("welch=True")
        if not hasattr(self, "global_fit_results"):
            self.global_fit_results = {}
        self.global_fit_results[(method, "; ".join(details))] = (
            self.global_fit_summary.copy()
        )
        return out

    def _global_fit_run(
        self,
        method="wald",
        split=None,
        covariate=None,
        welch=False,
        alpha=0.05,
        seed=None,
        split_prop=0.5,
        se_correction="kauermann-carroll",
        no_of_samples=500,
        p_method="chi2",
        mc_exceedances=None,
        unused_categories=None,
    ):
        """global_fit() itself (see its docstring); global_fit() records each result."""
        if method not in self._GLOBAL_FIT_METHODS:
            raise ValueError(
                f"method must be one of {self._GLOBAL_FIT_METHODS}, got {method!r}."
            )
        if unused_categories not in (None, "drop", "null"):
            raise ValueError(
                f"unused_categories must be None, 'drop' or 'null', got {unused_categories!r}."
            )
        if unused_categories == "null" and method != "andersen":
            raise ValueError("unused_categories='null' applies to method='andersen' only.")
        if covariate is not None:
            if method == "martin-lof":
                raise ValueError(
                    "covariate= is not used by method='martin-lof', which splits items, not "
                    "persons."
                )
            if split is None:
                split = "exogenous"
            elif split != "exogenous":
                raise ValueError(
                    f"covariate= is only used with split='exogenous', got split={split!r}. "
                    f"Pass split='exogenous' (or leave split out) to split by {covariate!r}."
                )
        if method == "martin-lof":
            if getattr(self, "estimation", None) != "cml":
                raise ValueError(
                    "method='martin-lof' needs conditional maximum likelihood (CML) "
                    "estimates. Re-run with calibrate(estimation='cml')."
                )
            if p_method not in ("chi2", "monte-carlo"):
                raise ValueError(
                    f"p_method must be 'chi2' or 'monte-carlo', got {p_method!r}."
                )
            if mc_exceedances is not None and int(mc_exceedances) < 1:
                raise ValueError(
                    f"mc_exceedances must be a positive integer or None, got {mc_exceedances!r}."
                )
            return self._martin_lof(
                "median" if split is None else split,
                alpha,
                p_method=p_method,
                no_of_samples=no_of_samples,
                seed=seed,
                mc_exceedances=mc_exceedances,
            )
        if split is None:
            split = "score"
        if split not in self._GLOBAL_FIT_SPLITS:
            raise ValueError(
                f"split must be one of {self._GLOBAL_FIT_SPLITS}, got {split!r}."
            )
        if not hasattr(self, "items"):
            self.calibrate()

        if method == "andersen":
            # Andersen's LR needs estimates that maximise the likelihood it
            # evaluates (CML). PAIR/CPAT estimates do not: under no misfit the
            # statistic floored at 0 on score/person-location splits, and on
            # exogenous splits was conservative for RSM/PCM, with a null
            # distribution that is not a scaled chi-square (RSM: 0 in 42-46%
            # of datasets), so no rescaling calibrates it.
            est = getattr(self, "estimation", None)
            if est not in ("cml", "cp-full-rcl"):
                raise ValueError(
                    "method='andersen' needs estimates that maximise the likelihood it "
                    "evaluates: calibrate(estimation='cml') or estimation='cp-full-rcl'. With "
                    f"estimation={est!r} the likelihood ratio is not calibrated (in simulation it "
                    "often floored at 0, and its null distribution was not a rescaled "
                    "chi-square). Use one of those, or method='wald' (the default), which is "
                    "calibrated for every split."
                )
            if not 0 < split_prop <= 0.5:
                raise ValueError(f"split_prop must be in (0, 0.5], got {split_prop!r}.")
            if est == "cp-full-rcl":
                if p_method not in ("chi2", "monte-carlo"):
                    raise ValueError(
                        f"p_method must be 'chi2' or 'monte-carlo', got {p_method!r}."
                    )
                if unused_categories is not None:
                    raise ValueError(
                        "unused_categories= applies to CML estimates only; 'cp-full-rcl' "
                        "estimates every threshold."
                    )
                if mc_exceedances is not None and int(mc_exceedances) < 1:
                    raise ValueError(
                        f"mc_exceedances must be a positive integer or None, got {mc_exceedances!r}."
                    )
                return self._composite_andersen(
                    split, covariate, seed, split_prop, alpha, p_method, no_of_samples, mc_exceedances
                )
            if p_method not in ("chi2", "monte-carlo", "auto"):
                raise ValueError(
                    f"p_method must be 'chi2' or 'monte-carlo', got {p_method!r}."
                )
            if mc_exceedances is not None and int(mc_exceedances) < 1:
                raise ValueError(
                    f"mc_exceedances must be a positive integer or None, got {mc_exceedances!r}."
                )
            self._cml_unused_fit = unused_categories
            try:
                return self._andersen_lr(
                    split,
                    covariate,
                    seed,
                    split_prop,
                    alpha,
                    p_method=p_method,
                    no_of_samples=no_of_samples,
                    mc_exceedances=mc_exceedances,
                )
            finally:
                self._cml_unused_fit = None

        if se_correction not in ("kauermann-carroll", None):
            raise ValueError(
                f"se_correction must be 'kauermann-carroll' or None, got {se_correction!r}."
            )
        if not 0 < split_prop <= 0.5:
            raise ValueError(f"split_prop must be in (0, 0.5], got {split_prop!r}.")
        groups = self._global_fit_groups(split, covariate, seed, split_prop)
        names = list(groups)
        fitted, parts = {}, []
        rng = np.random.default_rng(seed)
        boot = self._global_fit_needs_bootstrap()
        for name in names:
            m = self._global_fit_subset(groups[name])
            fitted[name] = m
            if boot:
                lab, est, _, ok, blk = self._global_fit_params(m, covariance=False)
                V = self._global_fit_bootstrap_cov(
                    groups[name], lab, no_of_samples, rng
                )
                parts.append((lab, est, V, ok, blk))
            else:
                parts.append(self._global_fit_params(m, se_correction is not None))
        if parts[0][0] != parts[1][0]:
            # the groups responded to different items: match parameters by name
            parts, blocks = self._global_fit_align(parts)
        else:
            blocks = parts[0][4]
        (lab, e1, V1, ok1, _), (_, e2, V2, ok2, _) = parts
        keep = ok1 & ok2
        if not keep.any():
            raise ValueError(
                "The two groups have no parameters estimable in both (e.g. they "
                "share no items), so there is nothing to compare."
            )
        idx = np.flatnonzero(keep)
        C = np.eye(len(idx))
        for blk in blocks:  # re-centre each location block over the kept entries
            pos = np.flatnonzero(np.isin(idx, blk))
            if len(pos):
                C[np.ix_(pos, pos)] -= 1.0 / len(pos)
        d = C @ (e1 - e2)[idx]
        S = C @ (V1 + V2)[np.ix_(idx, idx)] @ C.T
        W = float(d @ np.linalg.pinv(S, rcond=1e-10) @ d)
        df = int(np.linalg.matrix_rank(S, tol=1e-10 * np.abs(S).max()))
        if welch:  # deprecated previous reference (to be removed): df from the smaller group only
            ddf = max(min(len(groups[n]) for n in names) - df, 1)
            p = float(f_dist.sf(W / df, df, ddf))
        else:  # multivariate Welch (Nel & van der Merwe): each group's covariance from its own persons
            sub = np.ix_(idx, idx)
            nu = _welch_df(
                [C @ V1[sub] @ C.T, C @ V2[sub] @ C.T], [len(groups[n]) for n in names]
            )
            ddf = nu - df + 1
            p = _f_reference(W, df, nu + 1)
        se = np.sqrt(np.clip(np.diag(S), 0, None))
        kept_labels = [lab[i] for i in idx]
        self.global_fit_table = pd.DataFrame(
            {
                f"{names[0]}": (C @ e1[idx]),
                f"{names[1]}": (C @ e2[idx]),
                "Difference": d,
                "SE": se,
                "z": np.divide(d, se, out=np.full_like(d, np.nan), where=se > 0),
            },
            index=pd.Index(kept_labels, name="Parameter"),
        )
        self.global_fit_excluded = [lab[i] for i in np.flatnonzero(~keep)]
        self.global_fit_groups = fitted
        summary = {"method": method, "split": split, "statistic": W, "df": df}
        if split in ("score", "person_location"):
            summary["split_prop"] = split_prop
        summary["denominator df"] = ddf
        cml = getattr(self, "estimation", None) == "cml"
        cml_cov = (
            "hessian" if getattr(self, "se_method", None) == "hessian" else "sandwich"
        )
        summary["covariance"] = (
            "bootstrap" if boot else (cml_cov if cml else "sandwich")
        )
        summary["se_correction"] = None if (boot or (cml and cml_cov == "hessian")) else se_correction
        summary.update(
            {
                "p-value": p,
                "reject": bool(p <= alpha),
                "group sizes": {n: len(groups[n]) for n in names},
                "parameters excluded": len(self.global_fit_excluded),
            }
        )
        self.global_fit_summary = pd.Series(summary, name="Global fit")
        if self.global_fit_excluded:
            warnings.warn(
                f"{len(self.global_fit_excluded)} parameter(s) not compared because "
                f"they are not estimable in both groups (e.g. an item one group did not "
                f"respond to, or next to a score category unused in one group): "
                f"{self.global_fit_excluded[:10]}"
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
            'thresholds', each facet element's multiple locations
            (one per item, or one per threshold) are all plotted
            individually, named e.g. 'Rater_1_Item_1'/'Rater_1_1'. For
            'matrix', every (facet element, item, threshold) combination
            is plotted individually, named e.g. 'Rater_1_Item_1_1' --
            this can be a *lot* of individual values (facet elements x
            items x max_score), so facet_labels=True in particular can
            produce a very tall panel there. For 'bivector', its two
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
                grid, from summing its two marginal location vectors
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
            expands, <1 compresses; see each model's calibrate_*
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
            when item_level='items'. Also governs item_strip=True's
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
            one row per item, spanning its threshold range, divided
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
            lowest threshold. 'order' preserves self.items' index
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
            (facet_labels' binned panel, like item_labels' own,
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
            One-sided magnitude for persons' Count/Proportion axis
            (e.g. person_lim=30 shows 0 to 30). If None, chosen
            automatically from persons' natural peak, independently
            of items'/facets' scale. Persons' panel always keeps
            its full physical size regardless of this value -- items'
            and facets' panels (when they get a dedicated hist/kde
            panel, i.e. item_distribution=True and/or a non-facet_labels,
            non-overlaid facet panel) are sized proportionally against
            it, so a given Count/Proportion value reads at the same
            physical scale in every panel that participates.
        item_lim : float or None, default None
            One-sided magnitude for items' Count/Proportion axis.
            Same semantics as person_lim, for whichever panel items end
            up in (mirrored onto the same axis as persons when
            item_labels=False, or their item_distribution panel).
        facet_lim : float or None, default None
            One-sided magnitude for the facet-element distribution's
            Count/Proportion axis. Same semantics as person_lim, for
            whichever panel the facet distribution ends up in (overlaid
            onto the items panel/axis when overlay_facet=True, or its
            dedicated panel). Only used when the facet is shown as a
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
            Reduces the facet panel's physical size by this factor,
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
            needed -- the person distribution's size is never reduced
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
            than using matplotlib's automatic tick choice.
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
            "global",
            "items",
            "thresholds",
            "matrix",
            "bivector",
            "centrality",
            "pseudo_halo",
            "bistretch",
            "mixed",
        ):
            raise NotImplementedError(
                f"wright_map does not support the {model!r} MFRM model."
            )
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
        if (item_level == "thresholds" or item_strip) and not hasattr(
            self, "thresholds"
        ):
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
        own_thresholds = (
            self._thresholds_for(model)
            if has_facets
            else getattr(self, "thresholds", None)
        )
        base_items = own_items if item_names is None else own_items.loc[item_names]

        if has_facets:

            # facet_element_names always filters by rater name (matching
            # what it does for 'global'), applied before flattening below --
            # for 'items'/'thresholds' that's the raw table's row index;
            # for 'matrix' it's the first level of its (rater, item) row
            # index, which .loc[] with a plain list of rater names also
            # selects on
            def _filter_by_rater(table):
                return (
                    table
                    if facet_element_names is None
                    else table.loc[facet_element_names]
                )

            def _flatten_facets(table):
                # one row per rater, one column per item/threshold/category
                # -- stacking flattens that to one entry per combination,
                # e.g. 'Rater_1_Item_1' or 'Rater_1_1' for a 2-level table,
                # 'Rater_1_Item_1_1' for a 3-level one (matrix's (rater,
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
            # bivector's expansion below uses -- each model resolves it
            # to whatever shape (flat per-element vector here; a (facet
            # element, item, threshold) grid there) its restriction
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
                        "omega_thresholds": _filter_by_rater(
                            fe_stretch["omega_thresholds"]
                        ),
                    }
                else:
                    stretch_omega = {"omega": _filter_by_rater(fe_stretch["omega"])}
            elif model == "global":
                facets = _filter_by_rater(self.facet_effects_global)
            elif model == "mixed" and not expand_model:
                # each facet element's global effect (its mean effect over the item x threshold cells), whatever its
                # representation; expand_model=True shows every cell of the matrix layout instead
                sev = self.facet_effects_mixed.groupby(level=0, sort=False).mean().mean(axis=1)
                facets = _filter_by_rater(sev)
            elif model == "bivector" and not expand_model:
                # bivector's true parameters are the two marginal vectors
                # (per-item effects, per-threshold effects) its full
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
                # instance actually has: MFRM's rater-free softmax
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
                        cumsum = -_esf_lcw(thresh)
                        log_nums = cats * (theta - item_location) - cumsum
                        log_nums -= log_nums.max()
                        nums = np.exp(log_nums)
                        return nums[category] / nums.sum()

                    def exp_score_fn(theta, item_location, thresh):
                        cats = np.arange(len(thresh) + 1, dtype=float)
                        cumsum = -_esf_lcw(thresh)
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
                                return (
                                    sum(
                                        cat_prob_fn(theta, item_location, c, thresh)
                                        for c in range(k, m + 1)
                                    )
                                    - 0.5
                                )

                        else:  # half-point

                            def target(theta, k=k):
                                return exp_score_fn(theta, item_location, thresh) - (
                                    k - 0.5
                                )

                        display_matrix.loc[item, k] = brentq(target, lo, hi)
                matrix_for_points = display_matrix
            else:
                matrix_for_points = matrix

            # pandas' stack() stopped dropping NaN by default as of the
            # 2.1+ implementation, so drop the short items' padding NaNs
            # explicitly rather than relying on stack()'s own default
            stacked = matrix_for_points.stack().dropna()
            items = stacked.set_axis([f"{item} (τ{k})" for item, k in stacked.index])

            if item_strip:
                # kept grouped by item (not flattened) for the strip
                # renderer, which needs each item's own ordered threshold
                # list rather than one independent point per threshold.
                # Segments are the item's *most probable category*
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
                    T = -_esf_lcw(thresholds_natural)
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
                            else (intercept - hull[i - 1][1]) / (hull[i - 1][0] - slope)
                        )
                        seg_hi = (
                            None
                            if i == len(hull) - 1
                            else (hull[i + 1][1] - intercept) / (slope - hull[i + 1][0])
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
                    # geometrically (its probability "bump" never rises
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
                # item_strip's segment boundaries (built above, from
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

        # each distribution's natural peak, computed analytically
        # (no plotting needed) so panel sizing/axis ranges can be
        # resolved before the figure is even built. Computed
        # independently per distribution -- deliberately NOT sharing a
        # single "widest" value across persons/items/facets, since a
        # small-N facet's much more sharply peaked KDE would
        # otherwise force persons' well-populated distribution onto
        # a needlessly inflated scale (and vice versa for a tiny facet
        # squeezed onto persons' wide one)
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
        # ceiling (which previously left the axis's true top edge
        # short of its last drawn tick, e.g. ticks 0..28 against a
        # ceiling of 30). The 10% pad before searching guarantees the
        # chosen ceiling sits visibly above the true peak even when that
        # peak already IS a neat number itself (a small integer count,
        # for instance) -- without it, the tallest bar/curve would land
        # flush against the axis's edge, with no headroom at all.
        _nice_fractions = [1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10]
        _nice_divisors = {
            1: 4,
            1.5: 3,
            2: 4,
            2.5: 5,
            3: 3,
            4: 4,
            5: 5,
            6: 3,
            8: 4,
            10: 5,
        }

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
            item_range, item_step = _nice_ceil_and_step(
                item_natural_max, _integer_counts
            )
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
        # tick inside the margin and clear of row 0's text -- shared
        # so the two stay consistent if this value ever changes)
        item_row_margin = 0.45
        # same idea for the facet-labels panel, which never carries
        # distribution_markers' reserved band -- kept as its own named
        # constant (same value) purely for readability at its call sites
        facet_row_margin = 0.45

        # shared physical sizing for distribution_markers' ticks: a fixed
        # gap (inches) between a tick and its label, and a fixed tick
        # length (rows) on the item side -- both fixed in real inches
        # rather than a fraction of whatever data range is being plotted,
        # so the two sides (persons' count-axis-based scale vs items'
        # row-based scale) end up visually consistent instead of drifting
        # apart depending on the dataset
        mark_gap_in = 0.065
        mark_tick_rows = 0.15
        # physical breathing room (inches) between the widest mark's
        # label and where item names start -- separate from the +0.5-row
        # collision buffer below (that one exists purely to stop an
        # item's centred text reaching back into the mark's text; this
        # one is just visual spacing between the two blocks of text)
        item_label_gap_in = 0.08

        # when item_labels shows a dedicated item panel (rather than
        # item_distribution's separate histogram/KDE), distribution_markers'
        # item-side marks are drawn straight into that panel's row
        # grid, all at row 0 -- items then start uniformly at row 1,
        # regardless of which bin(s) the marks themselves land in.
        item_mark_rows = {}
        mark_row_depth = 0
        if (
            distribution_markers
            and item_labels
            and not item_distribution
            and not item_strip
        ):
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
                # threshold range. Row labels are drawn once per row,
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
                    ordered_names = sorted(
                        item_spans, key=lambda name: item_spans[name][0]
                    )
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
                # into row_height_in's measurement
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
                    person_panel_in = (
                        base_h if orientation == "vertical" else base_w
                    ) / 2
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
                        0,
                        0,
                        "μ+2σ",
                        rotation=90,
                        fontsize=labelsize,
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
                # into row_height_in's measurement (most visible on
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
            # budget they used to occupy -- and persons' panel always
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
            # The facet panel always matches items' effective scale.
            # When items have their own panel (item_labels=True, or
            # item_distribution=True), that's the same person_panel_h/
            # person_range reference item_dist_panel_h itself uses below.
            # When persons and items still mirror onto the *same* ax
            # instead (item_labels=False), items' effective scale IS ax's
            # own: a single physical axis has one px-per-unit throughout,
            # regardless of the two sides showing different data values,
            # so ax's combined span (person_range + item_range)
            # against its physical size gives that same reference
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
                        (
                            person_panel_h
                            if orientation == "vertical"
                            else person_panel_w
                        )
                        * (facet_range / person_range)
                        * person_scaling
                        / facet_scaling
                    )
                elif person_range + item_range * mirror_item_scale > 0:
                    # mirrored map: persons and items share ax, whose total
                    # span is (person_range + item_range * mirror_item_scale)
                    # -- the item side stretched by mirror_item_scale (==
                    # person_scaling here, 1 otherwise). Size the facet's
                    # panel so a facet count reads at the SAME physical
                    # per-count scale as an item count on that stretched item
                    # side: facet_panel_in / facet_range must equal
                    # mirror_item_scale * person_panel / total_span. (Reduces
                    # to the plain facet_range/(person_range+item_range) at
                    # mirror_item_scale=1; facet_scaling still overrides.)
                    facet_panel_in = (
                        (
                            person_panel_h
                            if orientation == "vertical"
                            else person_panel_w
                        )
                        * mirror_item_scale
                        * facet_range
                        / (person_range + item_range * mirror_item_scale)
                        / facet_scaling
                    )
                else:
                    facet_panel_in = (
                        base_h / 3 if orientation == "vertical" else base_w / 3
                    ) / facet_scaling

            # nothing needs its own panel only when there's no item panel
            # of any kind (neither a name-label panel nor a separate,
            # scale-matched distribution panel) *and* the facet
            # distribution is being overlaid onto the (then single,
            # mirrored) axis -- or, on a model with no facet structure at
            # all (SLM/PCM/RSM), there is nothing beyond persons/items to
            # share that axis with in the first place -- the two cases
            # that still collapse to a single simple axes, as in the
            # original two-family (persons/items) layout
            if (
                not item_labels
                and not item_dist_panel
                and (overlay_facet or not has_facets)
            ):
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
            # its duplicate tick labels)
            extra_axes = []
            if item_dist_panel:
                extra_axes.append(ax_item_dist)
            if item_labels:
                extra_axes.append(ax_items)
            if not overlay_facet and has_facets:
                extra_axes.append(ax_facet)
            outer_ax = extra_axes[-1] if extra_axes else ax
            interior_axes = extra_axes[:-1]

            # each later panel's opaque background would otherwise
            # paint over anything from an earlier (lower z-order) sibling
            # panel that legitimately overhangs across their shared
            # boundary -- e.g. distribution_markers' item marks, drawn
            # with clip_on=False so they aren't cut off at their
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
                # drawn -- their separate distribution panel if they
                # have one (even without item_labels, per item_dist_panel
                # above), the item_labels row panel otherwise (though
                # overlay_facet is already forced off whenever
                # item_labels=True, so this branch can't actually be
                # reached that way), or plain ax as the final fallback
                facet_dist_ax = (
                    ax_item_dist
                    if item_dist_panel
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
                    item_curve = (
                        item_sign
                        * mirror_item_scale
                        * (kde_items if prop else kde_items * no_of_items)
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
            # its panel does not (that panel has its own extent).
            facet_mirror_scale = mirror_item_scale if facet_dist_ax is ax else 1
            if facet_dist_ax is not None:
                if map_type == "hist":
                    facet_weights = (
                        facet_sign * facet_mirror_scale * np.ones_like(facets)
                    )
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
                    facet_curve = (
                        facet_sign
                        * facet_mirror_scale
                        * (kde_facets if prop else kde_facets * no_of_facets)
                    )

                    if overlay_facet:
                        if orientation == "vertical":
                            facet_dist_ax.plot(
                                x_grid,
                                facet_curve,
                                color=facet_color,
                                linewidth=line_width * 2,
                                label=facet_label,
                            )
                        else:
                            facet_dist_ax.plot(
                                facet_curve,
                                x_grid,
                                color=facet_color,
                                linewidth=line_width * 2,
                                label=facet_label,
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
            # top of either labelling style, so this is its if/elif
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
                            "#E69F00",
                            "#56B4E9",
                            "#009E73",
                            "#F0E442",
                            "#0072B2",
                            "#D55E00",
                            "#CC79A7",
                            (
                                "#FFFFFF"
                                if plot_style in self._DARK_BACKGROUND_STYLES
                                else "#000000"
                            ),
                        ]
                    elif shade == "dark":
                        color_map = (
                            sns.color_palette("dark", as_cmap=True)
                            if palette == "dark multi"
                            else sns.dark_palette(
                                base_color, reverse=True, as_cmap=True
                            )
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
                    # stop at, so stopping partway (at item_spans'
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
                        items_in_per_row = (items_bbox.width / fig.dpi) / abs(x1i - x0i)
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

            # ax's, ax_item_dist's, and ax_facet's Count/Proportion
            # limits are set explicitly further below (from
            # person_range/item_range/facet_range), which fully
            # supersedes pinning just one side of each to 0 here.

            padding = (hi - lo) * 0.05 if pad else 0
            loc_axis = (
                outer_ax if (has_extra_panels and orientation == "vertical") else ax
            )
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
                        axis="x",
                        bottom=False,
                        top=True,
                        labelbottom=False,
                        labeltop=True,
                    )
                else:
                    ax.spines["right"].set_visible(False)

            is_vertical = orientation == "vertical"

            # each panel's (lo, hi) limits, set directly from
            # person_range/item_range/facet_range (an explicit *_lim
            # override, or each distribution's natural peak,
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
                pairs |= {
                    (-t * neg_scale, -t) for t in _side_ticks(neg_reach, neg_step)
                }
                pairs = sorted(pairs)
                # item_dist's/facet's zero always sits exactly at the
                # boundary shared with its neighbour -- in horizontal
                # orientation that boundary is a narrow vertical seam
                # with both panels' "0" label sitting right next to
                # it, close enough to visually collide. Dropping the
                # secondary panel's redundant zero (ax's stays)
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
                    (
                        ("Density" if map_type == "kde" else "Proportion")
                        if prop
                        else "Count"
                    ),
                    fontsize=axis_font_size,
                    fontweight="bold",
                )
                if item_distribution:
                    # one shared "Count"/"Density"/"Proportion" label instead
                    # of one per panel -- re-centre ax's label (rather than also
                    # labelling ax_item_dist) across both panels' combined
                    # span, computed in ax's own axes-fraction coordinates
                    # since item_dist sits directly below it. The x offset
                    # has to clear whichever panel's tick labels are wider.
                    # Measuring the tick labels' rendered extent
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
                    (
                        ("Density" if map_type == "kde" else "Proportion")
                        if prop
                        else "Count"
                    ),
                    fontsize=axis_font_size,
                    fontweight="bold",
                )
                if item_distribution:
                    # one shared "Count"/"Density"/"Proportion" label
                    # instead of one per panel -- re-centre ax's label across both panels'
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

                    bottom_px = min(
                        _bottommost_tick_y(ax), _bottommost_tick_y(ax_item_dist)
                    )
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
                    for a in list(ax_items.patches)
                    + list(ax_items.texts)
                    + list(ax_items.lines)
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
                        gap_px = min(b.y0 for b in item_boxes) - max(
                            b.y1 for b in facet_boxes
                        )
                    else:
                        gap_px = min(b.x0 for b in facet_boxes) - max(
                            b.x1 for b in item_boxes
                        )
                    excess_in = gap_px / fig.dpi - facet_gap_in
                    if excess_in > 0:
                        facet_pos = ax_facet.get_position()
                        after_seam = [
                            a
                            for a in fig.axes
                            if (
                                a.get_position().y1 <= facet_pos.y1 + 1e-9
                                if is_vertical
                                else a.get_position().x0 >= facet_pos.x0 - 1e-9
                            )
                        ]
                        new_size = (
                            (fig_w_in, fig_h_in - excess_in)
                            if is_vertical
                            else (fig_w_in - excess_in, fig_h_in)
                        )
                        for a in fig.axes:
                            p = a.get_position()
                            if is_vertical:
                                y0_in = p.y0 * fig_h_in - (
                                    0 if a in after_seam else excess_in
                                )
                                a.set_position(
                                    [
                                        p.x0,
                                        y0_in / new_size[1],
                                        p.width,
                                        p.height * fig_h_in / new_size[1],
                                    ]
                                )
                            else:
                                x0_in = p.x0 * fig_w_in - (
                                    excess_in if a in after_seam else 0
                                )
                                a.set_position(
                                    [
                                        x0_in / new_size[0],
                                        p.y0,
                                        p.width * fig_w_in / new_size[0],
                                        p.height,
                                    ]
                                )
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
            # isn't a logit value (see the comment by stretch_omega's
            # definition above), so it gets its own axis/scale entirely,
            # centred on 1 rather than 0, one bar per rater in plain
            # facet-name order (unrelated to whatever order the shift
            # panel's binning happened to place them in). bistretch
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

                stretch_extra_w_in = (
                    stretch_gap_in + stretch_col_w_in + stretch_margin_in
                )
                stretch_new_w = stretch_orig_w + stretch_extra_w_in
                fig.set_size_inches(stretch_new_w, stretch_orig_h)
                stretch_shrink = stretch_orig_w / stretch_new_w
                for a in stretch_main_axes:
                    pos = a.get_position()
                    a.set_position(
                        [
                            pos.x0 * stretch_shrink,
                            pos.y0,
                            pos.width * stretch_shrink,
                            pos.height,
                        ]
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
                row_h = (
                    stretch_y1 - stretch_y0 - row_gap_frac * (n_stretch - 1)
                ) / n_stretch

                # Shared y-limits across every stretch panel (e.g.
                # bistretch's omega_items/omega_thresholds), so a bar's
                # length is comparable panel to panel rather than each
                # auto-scaling to its data independently.
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
                        (
                            (compress_color if v < 1 else expand_color)
                            if v == v
                            else "lightgray"
                        )
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
                        chosen = [0.0] + [
                            ticks[i][0] for i in range(k - 1, len(ticks), k)
                        ]
                        gaps = [
                            abs(_px(a) - _px(b)) for a, b in zip(chosen, chosen[1:])
                        ]
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
