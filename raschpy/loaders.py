import json
import warnings

import pandas as pd
import numpy as np

MAX_LISTED = 5  # values and items named in a loader warning


def _read_json(filename, index_col):
    """
    Read a JSON table: either a list of records (one object per row, e.g.
    DataFrame.to_json(orient='records')) or an object of columns
    ({column: {row label: value}}, DataFrame.to_json()'s default), or the
    'split' layout. index_col as for pandas.read_csv: for a list of records it
    names the position(s) of the identifier column(s); an object of columns
    already labels its rows, and these labels are kept unless index_col is None.
    """
    with open(filename, encoding="utf-8") as file:
        data = json.load(file)
    positions = [index_col] if isinstance(index_col, int) else index_col
    if isinstance(data, list):
        table = pd.DataFrame(data)
        if positions is not None:
            table = table.set_index([table.columns[i] for i in positions])
        return table
    if isinstance(data, dict) and {"columns", "data"} <= set(data):
        table = pd.DataFrame(data["data"], columns=data["columns"])
        if "index" not in data:  # rows unlabelled: read like a list of records
            if positions is not None:
                table = table.set_index([table.columns[i] for i in positions])
            return table
        table.index = data["index"]
    elif isinstance(data, dict):
        table = pd.DataFrame(data)
    else:
        raise ValueError(
            f"{filename}: expected a list of records or an object of columns."
        )
    if positions is None:
        return table.reset_index(drop=True)
    if len(positions) > 1:
        raise ValueError(
            f"{filename}: a JSON file with more than one identifier column must "
            "be a list of records, one object per row."
        )
    return table


def _read_file(filename, header, index_col, sheet_name=0, sep=","):
    """Shared file reader supporting CSV (or other delimited text), Excel and JSON."""
    f = filename.lower()
    if f.endswith(".xlsx"):
        return pd.read_excel(
            filename,
            sheet_name=sheet_name,
            header=header,
            index_col=index_col,
            engine="openpyxl",
        )
    elif f.endswith(".json"):
        return _read_json(filename, index_col)
    else:
        if sep == "," and f.endswith(".tsv"):
            sep = "\t"
        return pd.read_csv(
            filename,
            sep=sep,
            engine="python" if sep is None else "c",
            header=header,
            index_col=index_col,
        )


def _apply_default_names(responses, item_names, person_names):
    """Assign default item/person names when not provided by the file."""
    n_persons, n_items = responses.shape
    if not item_names:
        responses.columns = [f"Item_{i + 1}" for i in range(n_items)]
    if not person_names:
        responses.index = [f"Person_{i + 1}" for i in range(n_persons)]
    return responses


def _mark_missing(data, missing):
    """
    Set the missing-value codes in missing to NaN (data: a DataFrame or
    Series). A code matches however it was read: 99 matches the number 99 or
    99.0 and the text '99' or '99.0'.
    """
    if missing is None:
        return data
    codes = [missing] if isinstance(missing, (str, int, float)) else list(missing)
    numbers = []
    for code in codes:
        try:
            numbers.append(float(code))
        except (TypeError, ValueError):
            continue
    as_numbers = (
        data.apply(pd.to_numeric, errors="coerce")
        if isinstance(data, pd.DataFrame)
        else pd.to_numeric(data, errors="coerce")
    )
    texts = [str(code) for code in codes]
    return data.mask(data.isin(codes + texts) | as_numbers.isin(numbers))


def _entries(values, mask, items):
    """The (item, value) pairs of the cells in mask."""
    rows, cols = np.nonzero(np.asarray(mask))
    return pd.DataFrame(
        {"item": np.asarray(items)[cols], "value": np.asarray(values)[rows, cols]}
    )


def _coerce(data):
    """(numbers, the (item, value) pairs of the cells that are not numbers)."""
    numbers = data.apply(pd.to_numeric, errors="coerce")
    text = _entries(data.to_numpy(dtype=object), data.notna() & numbers.isna(),
                    data.columns)
    return numbers, text


def _pivot_long(raw, index_names, missing):
    """
    Turn long-format data (columns index_names + 'Item' + 'Score', one row per
    response) into a table of responses with index_names as the index, items
    as columns. Returns (responses, the (item, value) pairs of scores that are
    not numbers). Warns when a (person, item) pair has more than one score,
    keeping the lowest.
    """
    required = list(index_names) + ["Item", "Score"]
    absent = [c for c in required if c not in raw.columns]
    if absent:
        raise ValueError(
            f"Long-format data needs the columns {required}; missing: {absent}."
        )
    raw = raw[required].copy()
    raw["Score"] = _mark_missing(raw["Score"], missing)
    scores = pd.to_numeric(raw["Score"], errors="coerce")
    is_text = raw["Score"].notna() & scores.isna()
    text = pd.DataFrame(
        {"item": raw.loc[is_text, "Item"].to_numpy(),
         "value": raw.loc[is_text, "Score"].to_numpy()}
    )
    raw["Score"] = scores
    scored = raw[raw["Score"].notna()]
    repeated = scored.duplicated(list(index_names) + ["Item"], keep=False)
    if repeated.any():
        pairs = scored[repeated].drop_duplicates(list(index_names) + ["Item"])
        warnings.warn(
            f"{len(pairs)} response(s) appear more than once in the long-format "
            f"data with a score each (e.g. {_listed(pairs.iloc[0].tolist()[:-1])}); "
            "the lowest score of each was kept.",
            UserWarning,
            stacklevel=4,
        )
    responses = pd.pivot_table(
        raw,
        values="Score",
        index=list(index_names),
        columns="Item",
        aggfunc="min",
        dropna=False,
    )
    responses.columns.name = None
    return responses, text


def _listed(names):
    names = [_shown(n) for n in names]
    shown = ", ".join(names[:MAX_LISTED])
    if len(names) > MAX_LISTED:
        shown += f" and {len(names) - MAX_LISTED} more"
    return shown


def _shown(value):
    if isinstance(value, str):
        return value
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value)


def _warn_invalid(entries):
    """One UserWarning naming the values set to missing for not being valid scores."""
    if entries.empty:
        return
    parts = []
    counts = entries.groupby("value", sort=False)["item"]
    for value, items in sorted(counts, key=lambda group: -len(group[1])):
        shown = repr(value) if isinstance(value, str) else _shown(value)
        unique = list(dict.fromkeys(items))
        word = "item" if len(unique) == 1 else "items"
        parts.append(f"{shown} ({len(items)} cell(s); {word} {_listed(unique)})")
    listed = "; ".join(parts[:MAX_LISTED])
    if len(parts) > MAX_LISTED:
        listed += f"; and {len(parts) - MAX_LISTED} more value(s)"
    warnings.warn(
        f"{len(entries)} value(s) are not valid scores (whole numbers from 0 to "
        f"the maximum score) and were set to missing: {listed}. Pass missing= to "
        "mark missing-value codes.",
        UserWarning,
        stacklevel=4,
    )


def _whole(responses):
    """True where a response is a whole number of 0 or more."""
    return (responses % 1 == 0) & (responses >= 0)


def _finish(responses, valid, text):
    """
    Set the responses that are not valid scores to missing, warning once with
    the values affected, and split into valid and invalid (all-missing) rows.
    """
    bad = _entries(responses.to_numpy(), responses.notna() & ~valid, responses.columns)
    _warn_invalid(pd.concat([e for e in (text, bad) if not e.empty] or [bad],
                            ignore_index=True))
    return _split_valid_invalid(responses.where(valid).astype(float))


def _split_valid_invalid(responses):
    """
    Split responses into valid (at least one non-NaN) and invalid (all-NaN) rows.
    Returns (valid, invalid) DataFrames.
    """
    all_nan = responses.isnull().all(axis=1)
    return responses[~all_nan].copy(), responses[all_nan].copy()


def _load_persons_by_items(
    filename, item_names, person_names, long, missing, sheet_name, sep
):
    """
    Shared SLM/PCM/RSM reading: (responses as numbers, persons in rows with the
    index named 'Person', items in columns; the (item, value) pairs of cells
    that are not numbers).
    """
    if long:
        raw = _read_file(filename, 0, None, sheet_name, sep)
        responses, text = _pivot_long(raw, ["Person"], missing)
        responses = _apply_default_names(responses, item_names, person_names)
    else:
        header = 0 if item_names else None
        index_col = 0 if person_names else None
        raw = _read_file(filename, header, index_col, sheet_name, sep)
        raw = _apply_default_names(raw, item_names, person_names)
        responses, text = _coerce(_mark_missing(raw, missing))
    responses.index.name = "Person"
    responses.columns.name = None
    return responses, text


def _load_facet_tables(tables, item_names, missing, long, facet_name):
    """
    Shared MFRM reading for one table per facet element ({element: table}):
    (responses as numbers with a (facet_name, 'Person') index covering every
    combination, items in columns; the (item, value) pairs of cells that are
    not numbers).
    """
    if long:
        raw = pd.concat(
            [table.assign(**{facet_name: element}) for element, table in tables.items()],
            ignore_index=True,
        )
        responses, text = _pivot_long(raw, [facet_name, "Person"], missing)
    else:
        numbers, texts = {}, []
        for element, table in tables.items():
            if not item_names:
                table.columns = [f"Item_{i + 1}" for i in range(table.shape[1])]
            numbers[element], text = _coerce(_mark_missing(table, missing))
            texts.append(text)
        responses = pd.concat(numbers.values(), keys=numbers.keys())
        text = pd.concat([t for t in texts if not t.empty] or texts[:1],
                         ignore_index=True)
    return _full_facet_grid(responses, facet_name), text


def _full_facet_grid(responses, facet_name):
    """Reindex to every (facet element, person) combination, filling with NaN."""
    responses.index = responses.index.set_names([facet_name, "Person"])
    full_idx = pd.MultiIndex.from_product(
        [responses.index.get_level_values(0).unique(),
         responses.index.get_level_values(1).unique()],
        names=[facet_name, "Person"],
    )
    responses = responses.reindex(full_idx)
    responses.columns.name = None
    return responses


def _mfrm_finish(responses, max_score, text, sort=True):
    if max_score is None:
        max_score = int(np.nan_to_num(responses.where(_whole(responses)).max().max()))
    valid = _whole(responses) & (responses <= max_score)
    valid, invalid = _finish(responses, valid, text)
    if sort:
        return valid.sort_index(), invalid.sort_index()
    return valid, invalid


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------

def loadup_slm(
    filename,
    item_names=True,
    person_names=True,
    long=False,
    missing=None,
    sheet_name=0,
    sep=",",
):
    """
    Load and validate response data for the Simple Logistic Model (SLM).

    Reads a response matrix from CSV, Excel (.xlsx) or JSON, coerces values
    to numeric, and replaces any value that is not 0 or 1 with NaN, with a
    UserWarning naming the values replaced. Splits the result into valid rows
    (at least one non-NaN) and invalid rows (all NaN).

    Parameters
    ----------
    filename : str
        Path to the data file. Format inferred from extension:
        '.xlsx' -> Excel; '.json' -> JSON (a list of records, one per
        person, or an object of columns); anything else -> CSV.
    item_names : bool, default True
        If True, first row is used as item names.
        If False, items are labelled 'Item_1', 'Item_2', etc.
    person_names : bool, default True
        If True, first column is used as person names.
        If False, persons are labelled 'Person_1', 'Person_2', etc.
    long : bool, default False
        If True, expects long format with 'Person', 'Item', 'Score' columns.
        A response given more than once keeps the lowest score, with a
        UserWarning.
    missing : str, int, float, list, or None, default None
        Value(s) to treat as missing (e.g. 99, -9 or '.'). A number matches
        however it is written in the file (99, 99.0 or '99').
    sheet_name : str or int, default 0
        Excel files only: the sheet to read, by name or position.
    sep : str or None, default ','
        Delimited text files only: the column separator. None detects it.
        A '.tsv' file is read with tabs unless sep is given.

    Returns
    -------
    responses : pandas.DataFrame
        Valid responses, shape (persons, items), index named 'Person'.
        Values are 0, 1, or NaN.
    invalid_responses : pandas.DataFrame
        Rows with entirely missing data, excluded from responses.
    """
    responses, text = _load_persons_by_items(
        filename, item_names, person_names, long, missing, sheet_name, sep
    )
    return _finish(responses, responses.isin([0, 1]), text)


def loadup_pcm(
    filename,
    max_score_vector=None,
    item_names=True,
    person_names=True,
    long=False,
    missing=None,
    sheet_name=0,
    sep=",",
):
    """
    Load and validate response data for the Partial Credit Model (PCM).

    Reads a response matrix, coerces values to numeric, and replaces any value
    outside [0, item max score] with NaN, with a UserWarning naming the values
    replaced. Splits into valid and invalid (all-NaN) rows.

    Parameters
    ----------
    filename : str
        Path to the data file (.xlsx, .json, or CSV).
    max_score_vector : list, numpy.array, pandas.Series, or None, default None
        Maximum possible score for each item, in column order. If None,
        inferred from the data (the highest whole-number score of 0 or more
        on each item). Supply explicitly if any item's maximum is never
        observed in the data.
    item_names : bool, default True
        If True, first row is used as item names.
        If False, items are labelled 'Item_1', 'Item_2', etc.
    person_names : bool, default True
        If True, first column is used as person names.
        If False, persons are labelled 'Person_1', 'Person_2', etc.
    long : bool, default False
        If True, expects long format with 'Person', 'Item', 'Score' columns.
        A response given more than once keeps the lowest score, with a
        UserWarning.
    missing : str, int, float, list, or None, default None
        Value(s) to treat as missing (e.g. 99, -9 or '.'). A number matches
        however it is written in the file (99, 99.0 or '99').
    sheet_name : str or int, default 0
        Excel files only: the sheet to read, by name or position.
    sep : str or None, default ','
        Delimited text files only: the column separator. None detects it.
        A '.tsv' file is read with tabs unless sep is given.

    Returns
    -------
    responses : pandas.DataFrame
        Valid responses, shape (persons, items), index named 'Person'. Values
        are non-negative integers up to max_score_vector[item], or NaN.
    invalid_responses : pandas.DataFrame
        Rows with entirely missing data.
    """
    responses, text = _load_persons_by_items(
        filename, item_names, person_names, long, missing, sheet_name, sep
    )
    if max_score_vector is None:
        max_score_vector = responses.where(_whole(responses)).max().fillna(0)
    elif not isinstance(max_score_vector, pd.Series):
        max_score_vector = pd.Series(max_score_vector, index=responses.columns)

    valid = _whole(responses) & responses.le(max_score_vector, axis=1)
    return _finish(responses, valid, text)


def loadup_rsm(
    filename,
    max_score=None,
    item_names=True,
    person_names=True,
    long=False,
    missing=None,
    sheet_name=0,
    sep=",",
):
    """
    Load and validate response data for the Rating Scale Model (RSM).

    Reads a response matrix, coerces values to numeric, and replaces any value
    outside [0, max_score] with NaN, with a UserWarning naming the values
    replaced. Splits into valid and invalid (all-NaN) rows.

    Parameters
    ----------
    filename : str
        Path to the data file (.xlsx, .json, or CSV).
    max_score : int or None, default None
        Maximum possible score (shared across all items). If None, inferred
        from the data (the highest whole-number score of 0 or more). Supply
        explicitly if the maximum is never observed.
    item_names : bool, default True
        If True, first row is used as item names.
        If False, items are labelled 'Item_1', 'Item_2', etc.
    person_names : bool, default True
        If True, first column is used as person names.
        If False, persons are labelled 'Person_1', 'Person_2', etc.
    long : bool, default False
        If True, expects long format with 'Person', 'Item', 'Score' columns.
        A response given more than once keeps the lowest score, with a
        UserWarning.
    missing : str, int, float, list, or None, default None
        Value(s) to treat as missing (e.g. 99, -9 or '.'). A number matches
        however it is written in the file (99, 99.0 or '99').
    sheet_name : str or int, default 0
        Excel files only: the sheet to read, by name or position.
    sep : str or None, default ','
        Delimited text files only: the column separator. None detects it.
        A '.tsv' file is read with tabs unless sep is given.

    Returns
    -------
    responses : pandas.DataFrame
        Valid responses, shape (persons, items), index named 'Person'. Values
        are integers in [0, max_score] or NaN.
    invalid_responses : pandas.DataFrame
        Rows with entirely missing data.
    """
    responses, text = _load_persons_by_items(
        filename, item_names, person_names, long, missing, sheet_name, sep
    )
    if max_score is None:
        max_score = int(np.nan_to_num(responses.where(_whole(responses)).max().max()))

    valid = _whole(responses) & (responses <= max_score)
    return _finish(responses, valid, text)


def loadup_mfrm_single(
    filename,
    max_score=None,
    item_names=True,
    long=False,
    missing=None,
    facet_name="Rater",
    sheet_name=0,
    sep=",",
):
    """
    Load MFRM data from a single file with a (facet element, Person)
    MultiIndex, where the first index level is the additional facet (e.g.
    raters).

    The file must have two index columns: the facet element (level 0) and
    Person (level 1). Reindexes to a full facet element x Person grid
    (filling absent combinations with NaN), validates scores, and splits into
    valid and invalid rows. Values that are not valid scores are set to NaN,
    with a UserWarning naming them.

    Parameters
    ----------
    filename : str
        Path to the data file (.xlsx, .json, or CSV). A JSON file must be a
        list of records, one per (facet element, person).
    max_score : int or None, default None
        Maximum possible score per item. If None, inferred from the data.
    item_names : bool, default True
        If True, first row is used as item names.
        If False, items are labelled 'Item_1', 'Item_2', etc.
    long : bool, default False
        If True, expects long format with facet_name, 'Person', 'Item',
        'Score' columns. A response given more than once keeps the lowest
        score, with a UserWarning.
    facet_name : str, default 'Rater'
        Name of the facet: the name of the first index level, and of the
        facet column in long format.
    missing : str, int, float, list, or None, default None
        Value(s) to treat as missing (e.g. 99, -9 or '.'). A number matches
        however it is written in the file (99, 99.0 or '99').
    sheet_name : str or int, default 0
        Excel files only: the sheet to read, by name or position.
    sep : str or None, default ','
        Delimited text files only: the column separator. None detects it.
        A '.tsv' file is read with tabs unless sep is given.

    Returns
    -------
    responses : pandas.DataFrame
        Valid responses with a (facet_name, 'Person') MultiIndex and items as
        columns. Values are integers in [0, max_score] or NaN.
    invalid_responses : pandas.DataFrame
        Rows with entirely missing data.
    """
    if long:
        raw = _read_file(filename, 0, None, sheet_name, sep)
        responses, text = _pivot_long(raw, [facet_name, "Person"], missing)
    else:
        header = 0 if item_names else None
        raw = _read_file(filename, header, [0, 1], sheet_name, sep)
        if not item_names:
            raw.columns = [f"Item_{i + 1}" for i in range(raw.shape[1])]
        responses, text = _coerce(_mark_missing(raw, missing))
    if long and not item_names:
        responses.columns = [f"Item_{i + 1}" for i in range(responses.shape[1])]
    return _mfrm_finish(
        _full_facet_grid(responses, facet_name), max_score, text, sort=False
    )


def loadup_mfrm_xlsx_tabs(
    filename, max_score, item_names=True, missing=None, long=False, facet_name="Rater"
):
    """
    Load MFRM data from multiple sheets of a single Excel workbook.

    Each sheet corresponds to one facet element (e.g. rater); sheet names
    become the facet element identifiers. Merges all sheets into a
    (facet element, Person) MultiIndex DataFrame, validates scores, and
    splits into valid and invalid rows.

    Parameters
    ----------
    filename : str
        Path to the .xlsx file.
    max_score : int
        Maximum possible score per item. Values that are not whole numbers in
        [0, max_score] become NaN, with a UserWarning naming them.
    item_names : bool, default True
        If True, first row of each sheet is used as item names.
        If False, items are labelled 'Item_1', 'Item_2', etc.
    missing : str, int, float, list, or None, default None
        Value(s) to treat as missing (e.g. 99, -9 or '.'). A number matches
        however it is written in the file (99, 99.0 or '99').
    long : bool, default False
        If True, each sheet is in long format with 'Person', 'Item', 'Score'
        columns. A response given more than once keeps the lowest score,
        with a UserWarning.
    facet_name : str, default 'Rater'
        Name of the facet: the name of the first index level.

    Returns
    -------
    responses : pandas.DataFrame
        Valid responses with a (facet_name, 'Person') MultiIndex and items as
        columns.
    invalid_responses : pandas.DataFrame
        Rows with entirely missing data.
    """
    header = 0 if item_names or long else None
    sheets = pd.read_excel(
        filename,
        sheet_name=None,
        header=header,
        index_col=None if long else 0,
        engine="openpyxl",
    )
    responses, text = _load_facet_tables(sheets, item_names, missing, long, facet_name)
    return _mfrm_finish(responses, max_score, text)


def loadup_mfrm_multiple(
    filename_dict,
    max_score,
    item_names=True,
    missing=None,
    long=False,
    facet_name="Rater",
    sep=",",
):
    """
    Load MFRM data from multiple separate files, one per facet element (e.g.
    rater).

    Reads one file per facet element, merges into a (facet element, Person)
    MultiIndex DataFrame, validates scores, and splits into valid and
    invalid rows. Supports CSV, Excel (.xlsx), and JSON per facet element.

    Parameters
    ----------
    filename_dict : dict
        Mapping of {facet element name: filepath}. All files must have the
        same item structure.
    max_score : int
        Maximum possible score per item. Values that are not whole numbers in
        [0, max_score] become NaN, with a UserWarning naming them.
    item_names : bool, default True
        If True, first row of each file is used as item names.
        If False, items are labelled 'Item_1', 'Item_2', etc.
    missing : str, int, float, list, or None, default None
        Value(s) to treat as missing (e.g. 99, -9 or '.'). A number matches
        however it is written in the file (99, 99.0 or '99').
    long : bool, default False
        If True, each file is in long format with 'Person', 'Item', 'Score'
        columns. A response given more than once keeps the lowest score,
        with a UserWarning.
    facet_name : str, default 'Rater'
        Name of the facet: the name of the first index level.
    sep : str or None, default ','
        Delimited text files only: the column separator. None detects it.
        A '.tsv' file is read with tabs unless sep is given.

    Returns
    -------
    responses : pandas.DataFrame
        Valid responses with a (facet_name, 'Person') MultiIndex and items as
        columns.
    invalid_responses : pandas.DataFrame
        Rows with entirely missing data.
    """
    header = 0 if item_names or long else None
    tables = {
        element: _read_file(filename, header, None if long else 0, sep=sep)
        for element, filename in filename_dict.items()
    }
    responses, text = _load_facet_tables(tables, item_names, missing, long, facet_name)
    return _mfrm_finish(responses, max_score, text)


def loadup_exogenous(filename, person_names=True, sheet_name=0, sep=","):
    """
    Load person-level exogenous variables (e.g. Gender, Nationality or Age)
    for DIF analysis.

    Reads a person-indexed covariate table from CSV, Excel (.xlsx), or JSON.
    Unlike the response loaders, values are not coerced to numeric or
    validated against a score range — exogenous variables are typically
    categorical labels (e.g. 'Female', 'Chinese'), although they may be
    numerical (e.g. for Age) and are kept as supplied. Loaded independently
    of the response file rather than as extra columns in it, so a covariate
    never risks being parsed as an item score.

    Parameters
    ----------
    filename : str
        Path to the data file. Format inferred from extension:
        '.xlsx' -> Excel; '.json' -> JSON (a list of records, one per
        person, or an object of columns); anything else -> CSV.
    person_names : bool, default True
        If True, first column is used as person identifiers (the index).
        If False, persons are labelled 'Person_1', 'Person_2', etc., in
        file row order.
    sheet_name : str or int, default 0
        Excel files only: the sheet to read, by name or position.
    sep : str or None, default ','
        Delimited text files only: the column separator. None detects it.
        A '.tsv' file is read with tabs unless sep is given.

    Returns
    -------
    pandas.DataFrame
        Person-indexed covariate table, index named 'Person'. Pass directly
        as the `exogenous` argument to the SLM/PCM/RSM/MFRM constructors.
    """
    index_col = 0 if person_names else None
    exogenous = _read_file(filename, 0, index_col, sheet_name, sep)

    if not person_names:
        exogenous.index = [f"Person_{i + 1}" for i in range(exogenous.shape[0])]
    exogenous.index.name = "Person"

    return exogenous
