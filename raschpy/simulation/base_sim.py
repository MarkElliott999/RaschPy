import copy
import warnings

import numpy as np
import pandas as pd


class Rasch_Sim:
    """
    Abstract base class for RaschPy simulation objects.

    Provides shared rename utilities (items, persons), a random-number
    helper, and the chunked response/missing-data samplers used by all
    concrete simulation subclasses (SLM_Sim, RSM_Sim, PCM_Sim, MFRM_Sim).
    Not intended to be instantiated directly.
    """

    # Cells per row chunk when sampling (~32 MB per float64 working array),
    # bounding peak memory for large simulations. Chunking draws the same
    # uniforms in the same order as a single full-size draw, so results do
    # not depend on it.
    _CHUNK_CELLS = 1 << 22

    _MISSING_METHODS = ("bernoulli", "exact")

    def __init__(self):
        """Initialise the abstract base simulation object. Not intended for direct use."""
        pass

    def _check_missing_method(self, missing_method):
        """Validate and store the missing_method argument."""
        if missing_method not in self._MISSING_METHODS:
            raise ValueError(
                f"missing_method must be one of {self._MISSING_METHODS}, "
                f"got {missing_method!r}."
            )
        self.missing_method = missing_method

    def _row_chunks(self, n_rows):
        """Yield (start, stop) row ranges covering n_rows in chunks."""
        step = max(1, self._CHUNK_CELLS // max(1, self.no_of_items))
        for start in range(0, n_rows, step):
            yield start, min(start + step, n_rows)

    def _draw_scores(self, n_rows, cat_prob_rows):
        """
        Sample polytomous scores for n_rows stacked rows, chunk by chunk.

        cat_prob_rows(start, stop) returns [P(X=0), ..., P(X=K)] as arrays of
        shape (stop - start, no_of_items). Each score is the number of upper
        tail probabilities P(X>=c), c = 1..K, that its uniform draw falls
        below. Draws the same uniforms, in the same order, as a single
        self._rng.random((n_rows, no_of_items)) call.
        """
        scores = np.empty((n_rows, self.no_of_items), dtype=np.int64)
        for start, stop in self._row_chunks(n_rows):
            u = self._rng.random((stop - start, self.no_of_items))
            probs = cat_prob_rows(start, stop)
            chunk = np.zeros(u.shape, dtype=np.int64)
            for c in range(1, len(probs)):
                # left-to-right sum P(X=c) + ... + P(X=K)
                tail = probs[c].copy()
                for k in range(c + 1, len(probs)):
                    tail += probs[k]
                chunk += u < tail
            scores[start:stop] = chunk
        return scores

    def _missing_scan(self, rng, n_rows, window):
        """
        Draw one block's missing-data uniforms from rng, chunk by chunk.

        Returns the Bernoulli mask (u < missing) and the (values, flat
        indices) of draws within `window` of `missing`, for exact-count
        adjustment.
        """
        p = self.missing
        mask = np.empty((n_rows, self.no_of_items), dtype=bool)
        vals, idx = [], []
        for start, stop in self._row_chunks(n_rows):
            u = rng.random((stop - start, self.no_of_items))
            mask[start:stop] = u < p
            if window is not None:
                near = np.flatnonzero(np.abs(u - p) <= window)
                vals.append(u.ravel()[near])
                idx.append(near + start * self.no_of_items)
        if window is None:
            return mask, None, None
        return mask, np.concatenate(vals), np.concatenate(idx)

    def _missing_window(self, n):
        """
        Initial half-width around `missing` within which 'exact' collects
        candidate cells: about 6 SDs of the Bernoulli missing count, plus a
        margin, so a rescan with a wider window is very rarely needed.
        """
        p = self.missing
        return (6 * np.sqrt(n * p * (1 - p)) + 64) / n

    def _missing_block(self, n_rows):
        """
        Missing-data mask for one block of n_rows x no_of_items cells.

        'bernoulli': each cell is missing independently with probability
        self.missing (u < missing). 'exact': exactly round(missing * cells)
        cells are missing -- the cells with the smallest of the same
        uniforms -- so the pattern differs from 'bernoulli' only in the few
        cells whose uniforms lie nearest the cut-off. Both consume the same
        random draws, so the observed scores are identical either way.
        """
        if self.missing_method == "bernoulli":
            mask, _, _ = self._missing_scan(self._rng, n_rows, None)
            return mask

        p = self.missing
        n = n_rows * self.no_of_items
        k = int(np.floor(p * n + 0.5))
        start_rng = copy.deepcopy(self._rng)
        window = self._missing_window(n)
        mask, vals, idx = self._missing_scan(self._rng, n_rows, window)
        while True:
            excess = int(mask.sum()) - k
            if excess > 0:
                side = vals < p
            else:
                side = vals >= p
            if side.sum() >= abs(excess) or window >= 1:
                break
            # Too few candidates near the cut-off (very unlikely): regenerate
            # the same draws with a wider window.
            window = min(1.0, window * 4)
            _, vals, idx = self._missing_scan(copy.deepcopy(start_rng), n_rows, window)
        if excess != 0:
            v, i = vals[side], idx[side]
            order = np.lexsort((i, v))  # by value, ties by position
            flat = mask.reshape(-1)
            if excess > 0:
                flat[i[order[-excess:]]] = False  # drop largest below cut-off
            else:
                flat[i[order[:-excess]]] = True  # add smallest above cut-off
        return mask

    def _missing_mask(self, n_blocks=1, shared=True):
        """
        Missing-data mask for n_blocks stacked blocks of no_of_persons rows.

        shared=True draws one block's pattern and repeats it for every
        block; shared=False draws each block in turn ('exact' then holds
        per block). Returns None when missing == 0, drawing nothing.
        """
        if not self.missing:
            return None
        if shared or n_blocks == 1:
            mask = self._missing_block(self.no_of_persons)
            return np.tile(mask, (n_blocks, 1)) if n_blocks > 1 else mask
        return np.vstack([self._missing_block(self.no_of_persons) for _ in range(n_blocks)])

    def _responses_frame(self, scores, mask, index):
        """
        Wrap sampled scores as the responses DataFrame, applying missing data.

        Columns that receive a NaN become float64; columns with no missing
        cells stay int64 (matching pandas' masked-assignment upcasting).
        """
        if mask is None or not mask.any():
            return pd.DataFrame(scores, index=index, columns=self.item_names, copy=False)
        values = scores.astype(float)
        values[mask] = np.nan
        hit = mask.any(axis=0)
        if hit.all():
            return pd.DataFrame(values, index=index, columns=self.item_names, copy=False)
        return pd.DataFrame(
            {
                item: (values[:, j] if hit[j] else scores[:, j])
                for j, item in enumerate(self.item_names)
            },
            index=index,
        )

    @property
    def item_ids(self):
        """Alias for self.item_names."""
        return self.item_names

    @property
    def person_ids(self):
        """Alias for self.person_names."""
        return self.person_names

    def randoms(self):
        """
        Generate a (no_of_persons, no_of_items) array of uniform random numbers.

        Used internally by all simulation subclasses to sample response scores
        and apply missing data patterns. Draws from self._rng (seeded via the
        subclass's seed= constructor argument) rather than the numpy global
        random state, so simulations are reproducible when seed is set.

        Returns
        -------
        numpy.ndarray
            Shape (no_of_persons, no_of_items), values in [0, 1).
        """
        return self._rng.random((self.no_of_persons, self.no_of_items))

    def _name_sets(self):
        """Current item, person (and facet element) names, by kind."""
        sets = {"item": set(self.item_names), "person": set(self.person_names)}
        if hasattr(self, "facet_names"):
            sets["facet"] = set(self.facet_names)
        return sets

    def _relabel(self, kind, mapping):
        """
        Rename items, persons or facet elements everywhere they label the
        simulation: the responses, the generating parameters (Series,
        DataFrames, MultiIndex levels), the category-probability cache and
        the name lists. An axis is relabelled when all its labels are names
        of this kind (and not all names of another kind).
        """
        sets = self._name_sets()
        own = sets[kind]
        others = [s for k, s in sets.items() if k != kind]

        def is_axis(labels):
            try:
                labels = set(labels)
            except TypeError:
                return False
            return bool(labels) and labels <= own and not any(labels <= o for o in others)

        def f(x):
            return mapping.get(x, x)

        def map_index(idx):
            if isinstance(idx, pd.MultiIndex):
                return idx.set_levels([lvl.map(f) if is_axis(lvl) else lvl for lvl in idx.levels])
            return idx.map(f) if is_axis(idx) else idx

        def map_obj(o):
            if isinstance(o, pd.DataFrame):
                o = o.copy(deep=False)
                o.index, o.columns = map_index(o.index), map_index(o.columns)
                return o
            if isinstance(o, pd.Series):
                o = o.copy(deep=False)
                o.index = map_index(o.index)
                return o
            if isinstance(o, pd.Index):
                return map_index(o)
            if isinstance(o, list):
                return [f(x) for x in o] if is_axis(o) else o
            if isinstance(o, tuple):
                return tuple(map_obj(x) for x in o)
            if isinstance(o, dict):
                keys = {k: (f(k) if is_axis(o.keys()) else k) for k in o}
                return {keys[k]: map_obj(v) for k, v in o.items()}
            return o

        for name, value in list(vars(self).items()):
            new = map_obj(value)
            if new is not value:
                setattr(self, name, new)

    def _check_rename(self, label, current, old, new):
        """Warn and return False if renaming old -> new is not possible."""
        if old not in current:
            warnings.warn(f"Old {label} name {old!r} not found in data.", UserWarning, stacklevel=3)
        elif old == new:
            warnings.warn(f"New {label} name is the same as the old {label} name.", UserWarning, stacklevel=3)
        elif not isinstance(new, str):
            warnings.warn(f"{label.capitalize()} names must be strings.", UserWarning, stacklevel=3)
        elif new in current:
            warnings.warn(f"New {label} name is a duplicate of an existing {label} name.", UserWarning, stacklevel=3)
        else:
            return True
        return False

    def _check_rename_all(self, label, labels, current, new_names):
        """Warn and return False if new_names cannot replace current."""
        if len(new_names) != len(set(new_names)):
            warnings.warn(f"List of new {label} names contains duplicates.", UserWarning, stacklevel=3)
        elif len(new_names) != len(current):
            warnings.warn(
                f"Incorrect number of {label} names: {len(new_names)} provided, "
                f"{len(current)} {labels} in data.",
                UserWarning,
                stacklevel=3,
            )
        elif not all(isinstance(n, str) for n in new_names):
            warnings.warn(f"{label.capitalize()} names must be strings.", UserWarning, stacklevel=3)
        else:
            return True
        return False

    def rename_item(self, old, new):
        """
        Rename a single item in the simulation: in the responses and in the
        generating parameters, so recovery comparisons stay aligned.

        Parameters
        ----------
        old : str
            Current item name.
        new : str
            Desired new item name.
        """
        if self._check_rename("item", list(self.item_names), old, new):
            self._relabel("item", {old: new})

    def rename_items_all(self, new_names):
        """
        Rename all items at once, in the responses and in the generating
        parameters.

        Parameters
        ----------
        new_names : list of str
            New item names in the same order as self.item_names.
        """
        if self._check_rename_all("item", "items", list(self.item_names), new_names):
            self._relabel("item", dict(zip(self.item_names, new_names)))

    def rename_person(self, old, new):
        """
        Rename a single person in the simulation: in the responses and in the
        generating parameters, so recovery comparisons stay aligned.

        Parameters
        ----------
        old : str
            Current person name.
        new : str
            Desired new person name.
        """
        if self._check_rename("person", list(self.person_names), old, new):
            self._relabel("person", {old: new})

    def rename_persons_all(self, new_names):
        """
        Rename all persons at once, in the responses and in the generating
        parameters.

        Parameters
        ----------
        new_names : list of str
            New person names in the same order as self.person_names.
        """
        if self._check_rename_all("person", "persons", list(self.person_names), new_names):
            self._relabel("person", dict(zip(self.person_names, new_names)))

    def produce_df(self, rows, columns, row_names=None, column_names=None):
        """
        Build an empty MultiIndex DataFrame with the given row/column structure.

        Used internally by subclasses to construct structured output frames.

        Parameters
        ----------
        rows : list of iterables
            Factors for the row MultiIndex (passed to pd.MultiIndex.from_product).
        columns : list of iterables
            Factors for the column MultiIndex.
        row_names : list of str or None, default None
            Names for the row index levels.
        column_names : list of str or None, default None
            Names for the column index levels.

        Returns
        -------
        pandas.DataFrame
            Empty DataFrame with the specified MultiIndex structure.
        """

        row_index = pd.MultiIndex.from_product(rows, names=row_names)
        col_index = pd.MultiIndex.from_product(columns, names=column_names)

        return pd.DataFrame(index=row_index, columns=col_index)
