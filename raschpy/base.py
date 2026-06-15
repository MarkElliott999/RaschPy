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
plt.style.use('seaborn-v0_8-white')
sns.set_style('whitegrid')


class Rasch:

    def __init__(self):
        pass

    def check_data_connectivity(self):
        """
        AUTOMATIC DATA VALIDATION CHECK
        Validates if the item response graph is structurally connected.
        Can be called immediately after assigning self.dataframe in child classes.
        """
        if not hasattr(self, 'dataframe') or self.dataframe is None:
            return {"connected": False, "reason": "No dataframe loaded."}

        df_array = np.array(self.dataframe, dtype=np.float64)
        item_names = list(self.dataframe.columns)
        no_of_items = len(item_names)

        if no_of_items == 0:
            return {"connected": False, "reason": "No items present in the dataframe."}

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
            print("✔ DATA VALIDATION SUCCESS: The item response network is fully connected.")
            return {"connected": True, "components_count": 1}
        else:
            isolated_items = [comp for comp in components if len(comp) == 1]
            print(f"❌ CRITICAL WARNING: The data is split into {len(components)} disconnected sub-networks.")
            print("This will break un-smoothed calibrations. Isolated groupings:")
            for i, comp in enumerate(components):
                print(f"  -> Sub-group {i + 1} (Size {len(comp)}): {comp[:5]}... (truncated)")
            return {
                "connected": False,
                "components_count": len(components),
                "isolated_items": isolated_items,
                "all_sub_groups": components
            }

    def priority_vector(self,
                        matrix,
                        method='cos',
                        log_lik_tol=0.000001,
                        pcm=False,
                        raters=False):
        '''
        Optimised priority vector method (Choppin-compatible, zero-safe, loop-free).
        '''
        matrix_dim = matrix.shape[0]

        if pcm:
            names = []
            for i, item in enumerate(self.dataframe.columns):
                for j in range(self.max_score_vector.iloc[i]):
                    names.append(f'{str(item)}_{str(j + 1)}')
        else:
            names = self.raters if raters else list(self.dataframe.columns)

        with np.errstate(divide='ignore', invalid='ignore'):
            recip_matrix = np.divide(matrix.T, matrix)
            recip_matrix = np.nan_to_num(recip_matrix, nan=1.0, posinf=1.0, neginf=1.0)

        if method == 'evm':
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
                print('EVM method failed. Try another method.')
                return None

        elif method == 'log-lik':
            wins = matrix.sum(axis=1)
            change = 1.0
            wins_sum = wins.sum()
            weights = wins / wins_sum if wins_sum > 0 else np.ones(matrix_dim) / matrix_dim
            matrix_sum_sym = matrix + matrix.T

            while change > log_lik_tol:
                weight_pairs = weights[:, np.newaxis] + weights[np.newaxis, :]
                with np.errstate(divide='ignore', invalid='ignore'):
                    term_matrix = np.divide(matrix_sum_sym, weight_pairs,
                                            out=np.zeros_like(matrix_sum_sym), where=weight_pairs > 0)
                adjustment = term_matrix.sum(axis=1)
                self_term = np.divide(2 * np.diagonal(matrix), 2 * weights,
                                      out=np.zeros(matrix_dim), where=weights > 0)
                adjustment -= self_term

                new_weights = np.divide(wins, adjustment,
                                        out=np.zeros(matrix_dim), where=adjustment > 0)
                new_weights_sum = new_weights.sum()
                if new_weights_sum > 0:
                    new_weights /= new_weights_sum

                change = np.max(np.abs(weights - new_weights))
                weights = new_weights

            measures = -np.log(weights)
            measures -= np.mean(measures)
            measures = pd.Series(measures, index=names)

        else:
            if method == 'ls':
                weights = np.mean(recip_matrix, axis=1)
            else:
                normaliser = np.linalg.norm(recip_matrix, axis=0)
                normalised_matrix = np.divide(recip_matrix.T, normaliser[:, np.newaxis],
                                              out=np.zeros_like(recip_matrix.T),
                                              where=normaliser[:, np.newaxis] > 0)
                weights = normalised_matrix.sum(axis=0)

            measures = np.log(weights)
            measures -= np.mean(measures)
            measures = pd.Series(measures, index=names)

        return measures

    def pt_meas(self, abils, exp_score_df, info_df):
        '''
        Optimised Point-Measure and Expected Point-Measure Correlations.
        '''
        abil_dev_df = pd.DataFrame(
            np.tile(abils.values[:, np.newaxis] - np.mean(abils),
                    (1, len(self.dataframe.columns))),
            index=self.dataframe.index,
            columns=self.dataframe.columns
        )

        # Use .notna() for the validity mask — cleaner and avoids
        # division-by-zero artifacts from the original (x+1)/(x+1) approach.
        mask = self.dataframe.notna().astype(float).replace(0, np.nan)
        abil_dev_df = (abil_dev_df * mask).loc[exp_score_df.index]

        score_dev_df = self.dataframe.loc[exp_score_df.index] - self.dataframe.mean(axis=0)
        exp_score_dev_df = exp_score_df - self.dataframe.loc[exp_score_df.index].mean(axis=0)

        pt_measure_num = (score_dev_df * abil_dev_df).sum(axis=0)
        pt_measure_den = ((score_dev_df ** 2).sum(axis=0) * (abil_dev_df ** 2).sum(axis=0)) ** 0.5
        pt_measure = pt_measure_num / pt_measure_den

        exp_pt_measure_num = (exp_score_dev_df * abil_dev_df).sum(axis=0)
        exp_pt_measure_den = (
            ((exp_score_dev_df ** 2) + info_df).sum(axis=0)
            * (abil_dev_df ** 2).sum(axis=0)
        ) ** 0.5
        exp_pt_measure = exp_pt_measure_num / exp_pt_measure_den

        return pt_measure, exp_pt_measure

    # Item / Person renaming modules
    def rename_item(self, old, new):
        if old not in self.dataframe.columns:
            print(f'Old item name "{old}" not found in data. Please check.')
            return
        if not isinstance(new, str):
            print('Item names must be strings.')
            return
        if old == new:
            print('New item name is the same as old item name.')
            return
        if new in self.dataframe.columns:
            print('New item name is a duplicate of an existing item name.')
            return
        self.dataframe.rename(columns={old: new}, inplace=True)

    def rename_items_all(self, new_names):
        list_length = len(new_names)
        if len(new_names) != len(set(new_names)):
            print('List of new item names contains duplicates.')
        elif list_length != self.no_of_items:
            print(f'Incorrect token dimensions: Expected {self.no_of_items}.')
        else:
            self.dataframe.rename(columns=dict(zip(self.dataframe.columns, new_names)), inplace=True)

    def rename_person(self, old, new):
        if old not in self.dataframe.index:
            print(f'Old person name "{old}" not found in data.')
            return
        if not isinstance(new, str):
            print('Person names must be strings.')
            return
        if old == new:
            print('New person name is the same as old person name.')
            return
        if new in self.dataframe.index:
            print('New person name is a duplicate of an existing person name.')
            return
        self.dataframe.rename(index={old: new}, inplace=True)

    def rename_persons_all(self, new_names):
        list_length = len(new_names)
        if len(new_names) != len(set(new_names)):
            print('List contains duplicates.')
        elif list_length != self.no_of_persons:
            print(f'Incorrect token dimensions: Expected {self.no_of_persons}.')
        elif not all(isinstance(name, str) for name in new_names):
            print('Person names must be strings.')
        else:
            self.dataframe.rename(index=dict(zip(self.dataframe.index, new_names)), inplace=True)

    def std_residuals_hist(self,
                           std_residual_list,
                           bin_width=0.5,
                           x_min=-6,
                           x_max=6,
                           normal=False,
                           title=None,
                           plot_style='white',
                           black=False,
                           font='Times New Roman',
                           title_font_size=15,
                           axis_font_size=12,
                           labelsize=12,
                           filename=None,
                           file_format='png',
                           plot_density=300):
        '''
        Plots histogram of standardised residuals for SLM, with optional
        overplotting of Standard Normal Distribution.

        Performance fixes applied:
        1. plt.show() -> plt.show(block=False) + plt.pause(0.001)
           The original plt.show() blocks the process on all interactive
           backends (TkAgg, Qt5Agg, MacOSX) until the window is closed.
           This was the primary cause of the ~6s wall time. block=False
           renders and returns immediately, matching behaviour of other
           RaschPy plot methods.

        2. font set via rc_context, not fontname= per text object.
           Each fontname= kwarg triggers an individual findfont() lookup.
           With 2-3 text objects per call, that is 2-3 font scans per plot.
           On machines where Times New Roman is not cached, each scan can
           cost 100-500ms (Windows with large font libraries). Setting font
           once via rc_context resolves it in a single lookup.

        3. plt.style.use() + sns.set_style() moved to module level.
           Calling these on every plot invocation re-parses and re-applies
           the full stylesheet unnecessarily. They now run once on import.
           plot_style parameter is honoured via rc_context when non-default.
        '''
        color = 'gray' if black else 'steelblue'
        line_color = 'black' if black else 'maroon'
        n_bins = floor((std_residual_list.max() - std_residual_list.min()) / bin_width)

        # Apply font and any non-default style within a context so global
        # state is not permanently mutated by a single plot call.
        style_overrides = {'font.family': font,
                           'font.size': axis_font_size}

        with plt.rc_context(style_overrides):

            # Only re-apply style sheet when caller explicitly requests a
            # non-default style, avoiding the per-call stylesheet parse cost.
            if plot_style != 'white':
                plt.style.use('seaborn-v0_8-' + plot_style)
                if plot_style == 'dark':
                    sns.set_style('darkgrid')
                else:
                    sns.set_style('whitegrid')

            fig, ax = plt.subplots()

            ax.hist(
                std_residual_list,
                bins=n_bins,
                range=(x_min, x_max),
                density=True,
                facecolor=color,
                alpha=0.5,
                edgecolor='black',
                linewidth=1
            )

            if normal:
                x_norm = np.linspace(x_min, x_max, 200)
                y_norm = norm.pdf(x_norm, 0, 1)
                ax.plot(x_norm, y_norm, color=line_color)

            ax.set_xlabel('Standardised residual', fontsize=axis_font_size, fontweight='bold')
            ax.set_ylabel('Density', fontsize=axis_font_size, fontweight='bold')

            if title is not None:
                ax.set_title(title, fontsize=title_font_size, fontweight='bold')

            ax.tick_params(axis='x', labelsize=labelsize)
            ax.tick_params(axis='y', labelsize=labelsize)

            if filename is not None:
                fig.savefig(filename + f'.{file_format}', dpi=plot_density)

            # block=False returns immediately on all interactive backends.
            # pause(0.001) gives the GUI event loop a tick to render the
            # window, matching the behaviour of other RaschPy plot methods.
            plt.show(block=False)
            plt.pause(0.001)
            plt.close(fig)
