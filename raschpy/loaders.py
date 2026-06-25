import pandas as pd
import numpy as np


def _read_file(filename, header, index_col):
    """Shared file reader supporting CSV, Excel and JSON."""
    f = filename.lower()
    if f.endswith(".xlsx"):
        return pd.read_excel(
            filename,
            sheet_name=0,
            header=header,
            index_col=index_col,
            engine="openpyxl",
        )
    elif f.endswith(".json"):
        return pd.read_json(filename)
    else:
        return pd.read_csv(filename, header=header, index_col=index_col)


def _apply_default_names(responses, item_names, person_names):
    """Assign default item/person names when not provided by the file."""
    n_persons, n_items = responses.shape
    if not item_names:
        responses.columns = [f"Item_{i + 1}" for i in range(n_items)]
    if not person_names:
        responses.index = [f"Person_{i + 1}" for i in range(n_persons)]
    return responses


def _split_valid_invalid(responses):
    """
    Split responses into valid (at least one non-NaN) and invalid (all-NaN) rows.
    Returns (valid, invalid) DataFrames.
    The np.where(isna, nan, x) loops in the original were pure no-ops after
    .astype(float) — NaN is already float NaN. Removed entirely.
    """
    all_nan = responses.isnull().all(axis=1)
    return responses[~all_nan].copy(), responses[all_nan].copy()


def _mfrm_reindex_and_stack(responses_dict, item_ids, scores):
    """
    Shared MFRM finalisation: reindex to full (Rater x Person) grid,
    validate scores, then stack back to (Rater, Person) MultiIndex.

    Replaces the manual O(raters*persons) loop in the original with a single
    pd.reindex() call, then uses unstack/stack to produce the correct shape.
    """
    rater_names = list(responses_dict.keys())
    all_persons = np.unique(
        np.concatenate([df.index.to_numpy() for df in responses_dict.values()])
    )

    # Reindex each rater DataFrame to the full person set (fills missing with NaN)
    for rater in rater_names:
        responses_dict[rater] = (
            responses_dict[rater]
            .reindex(all_persons)
            .apply(pd.to_numeric, errors="coerce")
        )

    combined = pd.concat(responses_dict.values(), keys=rater_names)
    combined.index.names = ["Rater", "Person"]
    combined = combined.where(combined.isin(scores), np.nan).astype(float)

    # Pivot to (Person x (Item, Rater)) then back to (Rater, Person) x Item
    # so that all (Rater, Person) combinations are present
    full_idx = pd.MultiIndex.from_product(
        [rater_names, all_persons], names=["Rater", "Person"]
    )
    combined = combined.reindex(full_idx)

    valid, invalid = _split_valid_invalid(combined)
    return valid.sort_index(), invalid.sort_index()


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------


def loadup_slm(filename, item_names=True, person_names=True, long=False):
    """
    Load and validate response data for the Simple Logistic Model (SLM).

    Reads a response matrix from CSV, Excel (.xlsx), or JSON, coerces values
    to numeric, and replaces any value that is not 0 or 1 with NaN. Splits
    the result into valid rows (at least one non-NaN) and invalid rows (all NaN).

    Parameters
    ----------
    filename : str
        Path to the data file. Format inferred from extension:
        '.xlsx' -> Excel; '.json' -> JSON; anything else -> CSV.
    item_names : bool, default True
        If True, first row is used as item names.
        If False, items are labelled 'Item_1', 'Item_2', etc.
    person_names : bool, default True
        If True, first column is used as person names.
        If False, persons are labelled 'Person_1', 'Person_2', etc.
    long : bool, default False
        If True, expects long format with 'Person', 'Item', 'Score' columns.

    Returns
    -------
    responses : pandas.DataFrame
        Valid responses, shape (persons, items). Values are 0, 1, or NaN.
    invalid_responses : pandas.DataFrame
        Rows with entirely missing data, excluded from responses.
    """
    header = 0 if item_names else None
    index_col = 0 if person_names else None

    responses = _read_file(filename, header, index_col)

    if long:
        responses = pd.pivot_table(
            responses,
            values="Score",
            index="Person",
            columns="Item",
            aggfunc="min",
            dropna=False,
        )

    responses = _apply_default_names(responses, item_names, person_names)

    # Coerce to numeric then keep only valid dichotomous scores (0, 1)
    responses = responses.apply(pd.to_numeric, errors="coerce")
    responses = responses.where(responses.isin([0, 1]), np.nan).astype(float)

    return _split_valid_invalid(responses)


def loadup_pcm(
    filename, max_score_vector=None, item_names=True, person_names=True, long=False
):
    """
    Load and validate response data for the Partial Credit Model (PCM).

    Reads a response matrix, coerces values to numeric, and replaces any value
    that is not a non-negative integer at or below each item's maximum score
    with NaN. Splits into valid and invalid (all-NaN) rows.

    Parameters
    ----------
    filename : str
        Path to the data file (.xlsx, .json, or CSV).
    max_score_vector : list, pandas.Series, or None, default None
        Maximum possible score for each item, in column order. If None,
        the maximum observed value per column is used. Supply explicitly
        if any item's maximum is never observed in the data.
    item_names : bool, default True
        If True, first row is used as item names.
        If False, items are labelled 'Item_1', 'Item_2', etc.
    person_names : bool, default True
        If True, first column is used as person names.
        If False, persons are labelled 'Person_1', 'Person_2', etc.
    long : bool, default False
        If True, expects long format with 'Person', 'Item', 'Score' columns.

    Returns
    -------
    responses : pandas.DataFrame
        Valid responses, shape (persons, items). Values are non-negative integers
        up to max_score_vector[item], or NaN.
    invalid_responses : pandas.DataFrame
        Rows with entirely missing data.
    """
    header = 0 if item_names else None
    index_col = 0 if person_names else None

    responses = _read_file(filename, header, index_col)

    if long:
        responses = pd.pivot_table(
            responses,
            values="Score",
            index="Person",
            columns="Item",
            aggfunc="min",
            dropna=False,
        )

    responses = _apply_default_names(responses, item_names, person_names)
    responses = responses.apply(pd.to_numeric, errors="coerce")

    if max_score_vector is None:
        max_score_vector = responses.max()
    elif not isinstance(max_score_vector, pd.Series):
        max_score_vector = pd.Series(max_score_vector, index=responses.columns)

    # Vectorised validation: value must be a non-negative integer ≤ item max score
    # BUG FIX (original): max_score_vector computed before renaming but indexed by
    # integer position — fragile. Now indexed by column name after renaming.
    is_valid = (
        (responses % 1 == 0) & (responses >= 0) & responses.le(max_score_vector, axis=1)
    )
    responses = responses.where(is_valid, np.nan).astype(float)

    return _split_valid_invalid(responses)


def loadup_rsm(
    filename, max_score=None, item_names=True, person_names=True, long=False
):
    """
    Load and validate response data for the Rating Scale Model (RSM).

    Reads a response matrix, coerces values to numeric, and replaces any value
    outside [0, max_score] with NaN. Splits into valid and invalid (all-NaN) rows.

    Parameters
    ----------
    filename : str
        Path to the data file (.xlsx, .json, or CSV).
    max_score : int or None, default None
        Maximum possible score (shared across all items). If None, inferred
        from the data. Supply explicitly if the maximum is never observed.
    item_names : bool, default True
        If True, first row is used as item names.
        If False, items are labelled 'Item_1', 'Item_2', etc.
    person_names : bool, default True
        If True, first column is used as person names.
        If False, persons are labelled 'Person_1', 'Person_2', etc.
    long : bool, default False
        If True, expects long format with 'Person', 'Item', 'Score' columns.

    Returns
    -------
    responses : pandas.DataFrame
        Valid responses, shape (persons, items). Values are integers in
        [0, max_score] or NaN.
    invalid_responses : pandas.DataFrame
        Rows with entirely missing data.
    """
    header = 0 if item_names else None
    index_col = 0 if person_names else None

    responses = _read_file(filename, header, index_col)

    if long:
        responses = pd.pivot_table(
            responses,
            values="Score",
            index="Person",
            columns="Item",
            aggfunc="min",
            dropna=False,
        )

    responses = _apply_default_names(responses, item_names, person_names)
    responses = responses.apply(pd.to_numeric, errors="coerce")

    if max_score is None:
        max_score = int(responses.max().max())

    is_valid = (responses % 1 == 0) & (responses >= 0) & (responses <= max_score)
    responses = responses.where(is_valid, np.nan).astype(float)

    return _split_valid_invalid(responses)


def loadup_mfrm_single(filename, max_score=None, item_names=True, long=False):
    """
    Load MFRM data from a single file with a (Rater, Person) MultiIndex.

    The file must have two index columns: Rater (level 0) and Person (level 1).
    Reindexes to a full Rater x Person grid (filling absent combinations with
    NaN), validates scores, and splits into valid and invalid rows.

    Parameters
    ----------
    filename : str
        Path to the data file (.xlsx or CSV).
    max_score : int or None, default None
        Maximum possible score per item. If None, inferred from the data.
    item_names : bool, default True
        If True, first row is used as item names.
        If False, items are labelled 'Item_1', 'Item_2', etc.
    long : bool, default False
        If True, expects long format with 'Rater', 'Person', 'Item', 'Score' columns.

    Returns
    -------
    responses : pandas.DataFrame
        Valid responses with (Rater, Person) MultiIndex and items as columns.
        Values are integers in [0, max_score] or NaN.
    invalid_responses : pandas.DataFrame
        Rows with entirely missing data.
    """
    header = 0 if item_names else None
    f = filename.lower()

    if f.endswith(".xlsx"):
        responses = pd.read_excel(
            filename, sheet_name=0, header=header, index_col=[0, 1], engine="openpyxl"
        )
    else:
        responses = pd.read_csv(filename, header=header, index_col=[0, 1])

    if long:
        responses = pd.pivot_table(
            responses,
            values="Score",
            index=["Rater", "Person"],
            columns="Item",
            aggfunc="min",
            dropna=False,
        )

    n_items = responses.shape[1]
    if not item_names:
        responses.columns = [f"Item_{i + 1}" for i in range(n_items)]

    responses = responses.apply(pd.to_numeric, errors="coerce")

    if max_score is None:
        max_score = int(responses.max().max())
    scores = np.arange(max_score + 1)

    # BUG FIX (original): manual O(raters*persons) loop replaced with reindex
    rater_names = responses.index.get_level_values(0).unique()
    persons = responses.index.get_level_values(1).unique()
    full_idx = pd.MultiIndex.from_product(
        [rater_names, persons], names=["Rater", "Person"]
    )
    responses = responses.reindex(full_idx)
    responses = responses.where(responses.isin(scores), np.nan).astype(float)

    return _split_valid_invalid(responses)


def loadup_mfrm_xlsx_tabs(
    filename, max_score, item_names=True, missing=None, long=False
):
    """
    Load MFRM data from multiple sheets of a single Excel workbook.

    Each sheet corresponds to one rater; sheet names become rater identifiers.
    Merges all sheets into a (Rater, Person) MultiIndex DataFrame, validates
    scores, and splits into valid and invalid rows.

    Parameters
    ----------
    filename : str
        Path to the .xlsx file.
    max_score : int
        Maximum possible score per item. Values outside [0, max_score] become NaN.
    item_names : bool, default True
        If True, first row of each sheet is used as item names.
        If False, items are labelled 'Item_1', 'Item_2', etc.
    missing : str, int, float, list, or None, default None
        Value(s) in the raw data to treat as missing (NaN) before coercion.
        Useful for codes such as -99 or 'M'.
    long : bool, default False
        If True, each sheet is in long format with 'Rater', 'Person', 'Item',
        'Score' columns.

    Returns
    -------
    responses : pandas.DataFrame
        Valid responses with (Rater, Person) MultiIndex and items as columns.
    invalid_responses : pandas.DataFrame
        Rows with entirely missing data.
    """
    header = 0 if item_names else None

    sheets = pd.read_excel(
        filename, sheet_name=None, header=header, index_col=0, engine="openpyxl"
    )

    if long:
        combined = pd.concat(sheets.values(), keys=sheets.keys())
        combined.index.names = ["Rater", "Person"]
        combined = pd.pivot_table(
            combined,
            values="Score",
            index=["Rater", "Person"],
            columns="Item",
            aggfunc="min",
            dropna=False,
        )
        sheets = {
            rater: combined.xs(rater)
            for rater in combined.index.get_level_values(0).unique()
        }

    n_items = next(iter(sheets.values())).shape[1]
    if not item_names:
        item_ids = [f"Item_{i + 1}" for i in range(n_items)]
        for df in sheets.values():
            df.columns = item_ids

    scores = np.arange(max_score + 1)

    # Apply missing value substitution before numeric coercion
    # BUG FIX (original): missing= was ignored entirely in this function
    for rater, df in sheets.items():
        if missing is not None:
            missing_list = (
                [missing] if isinstance(missing, (str, int, float)) else list(missing)
            )
            df = df.replace(missing_list, np.nan)
            sheets[rater] = df
        # Floor to integer categories (MFRM convention matches original behaviour)
        sheets[rater] = np.floor(df.apply(pd.to_numeric, errors="coerce"))

    return _mfrm_reindex_and_stack(sheets, None, scores)


def loadup_mfrm_multiple(
    filename_dict, max_score, item_names=True, missing=None, long=False
):
    """
    Load MFRM data from multiple separate files, one per rater.

    Reads one file per rater, merges into a (Rater, Person) MultiIndex
    DataFrame, validates scores, and splits into valid and invalid rows.
    Supports CSV, Excel (.xlsx), and JSON per rater.

    Parameters
    ----------
    filename_dict : dict
        Mapping of {rater_name: filepath}. All files must have the same
        item structure.
    max_score : int
        Maximum possible score per item. Values outside [0, max_score] become NaN.
    item_names : bool, default True
        If True, first row of each file is used as item names.
        If False, items are labelled 'Item_1', 'Item_2', etc.
    missing : str, int, float, list, or None, default None
        Value(s) to treat as missing before numeric coercion.
    long : bool, default False
        If True, each file is in long format with 'Person', 'Item', 'Score' columns.

    Returns
    -------
    responses : pandas.DataFrame
        Valid responses with (Rater, Person) MultiIndex and items as columns.
    invalid_responses : pandas.DataFrame
        Rows with entirely missing data.
    """
    header = 0 if item_names else None

    sheets = {}
    for rater, filename in filename_dict.items():
        sheets[rater] = _read_file(filename, header=header, index_col=0)

    if long:
        combined = pd.concat(sheets.values(), keys=sheets.keys())
        combined.index.names = ["Rater", "Person"]
        combined = pd.pivot_table(
            combined,
            values="Score",
            index=["Rater", "Person"],
            columns="Item",
            aggfunc="min",
            dropna=False,
        )
        sheets = {
            rater: combined.xs(rater)
            for rater in combined.index.get_level_values(0).unique()
        }

    n_items = next(iter(sheets.values())).shape[1]
    if not item_names:
        item_ids = [f"Item_{i + 1}" for i in range(n_items)]
        for df in sheets.values():
            df.columns = item_ids

    scores = np.arange(max_score + 1)

    # Apply missing value substitution before numeric coercion
    # BUG FIX (original): loop body referenced responses.columns (dict, no .columns)
    for rater, df in sheets.items():
        if missing is not None:
            missing_list = (
                [missing] if isinstance(missing, (str, int, float)) else list(missing)
            )
            df = df.replace(missing_list, np.nan)
        sheets[rater] = np.floor(df.apply(pd.to_numeric, errors="coerce"))

    return _mfrm_reindex_and_stack(sheets, None, scores)
