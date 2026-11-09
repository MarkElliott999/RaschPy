"""Tests for raschpy.loaders: file formats, missing-value codes, the warning naming
values that are not valid scores, long format, and index naming."""
import json
import warnings

import numpy as np
import pandas as pd
import pytest

from raschpy import (
    loadup_exogenous,
    loadup_mfrm_multiple,
    loadup_mfrm_single,
    loadup_mfrm_xlsx_tabs,
    loadup_pcm,
    loadup_rsm,
    loadup_slm,
)

WIDE = pd.DataFrame(
    {"ID": ["p1", "p2", "p3"], "Q1": [1, 0, 1], "Q2": [0, 1, 1]}
)


def write(tmp_path, name, frame, **kwargs):
    path = tmp_path / name
    if name.endswith(".xlsx"):
        frame.to_excel(path, index=False, **kwargs)
    elif name.endswith(".json"):
        frame.to_json(path, **kwargs)
    else:
        frame.to_csv(path, index=False, **kwargs)
    return str(path)


def load_quietly(loader, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a clean file loads without warnings
        return loader(*args, **kwargs)


def expected_wide():
    expected = WIDE.set_index("ID").astype(float)
    expected.index.name = "Person"
    return expected


@pytest.mark.parametrize(
    "name, kwargs",
    [
        ("data.csv", {}),
        ("data.xlsx", {}),
        ("data.tsv", {"sep": "\t"}),
        ("records.json", {"orient": "records"}),
        ("split.json", {"orient": "split", "index": False}),
    ],
)
def test_formats_give_the_same_responses(tmp_path, name, kwargs):
    path = write(tmp_path, name, WIDE, **kwargs)
    responses, invalid = load_quietly(loadup_slm, path)
    pd.testing.assert_frame_equal(responses, expected_wide())
    assert invalid.empty


def test_json_object_of_columns(tmp_path):
    path = tmp_path / "columns.json"
    WIDE.set_index("ID").to_json(path)
    responses, _ = load_quietly(loadup_slm, str(path))
    pd.testing.assert_frame_equal(responses, expected_wide())
    numbered, _ = load_quietly(loadup_slm, str(path), person_names=False)
    assert list(numbered.index) == ["Person_1", "Person_2", "Person_3"]


def test_json_records_without_person_names(tmp_path):
    path = write(tmp_path, "records.json", WIDE.drop(columns="ID"), orient="records")
    responses, _ = load_quietly(loadup_slm, path, person_names=False)
    assert list(responses.index) == ["Person_1", "Person_2", "Person_3"]
    assert list(responses.columns) == ["Q1", "Q2"]


def test_separator_choice_and_detection(tmp_path):
    path = write(tmp_path, "data.txt", WIDE, sep=";")
    for sep in (";", None):
        responses, _ = load_quietly(loadup_slm, path, sep=sep)
        pd.testing.assert_frame_equal(responses, expected_wide())


def test_excel_sheet_choice(tmp_path):
    path = tmp_path / "book.xlsx"
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame({"note": ["not data"]}).to_excel(writer, sheet_name="Notes", index=False)
        WIDE.to_excel(writer, sheet_name="Responses", index=False)
    responses, _ = load_quietly(loadup_slm, str(path), sheet_name="Responses")
    pd.testing.assert_frame_equal(responses, expected_wide())


def test_index_is_named_person_without_person_names(tmp_path):
    path = write(tmp_path, "data.csv", WIDE.drop(columns="ID"))
    responses, _ = load_quietly(loadup_slm, path, person_names=False)
    assert responses.index.name == "Person"


def test_missing_codes_match_however_written(tmp_path):
    frame = pd.DataFrame(
        {"ID": ["p1", "p2", "p3", "p4"], "Q1": ["2", "99", "1", "0"],
         "Q2": ["1", "99.0", ".", "2"]}
    )
    path = write(tmp_path, "data.csv", frame)
    responses, invalid = load_quietly(loadup_pcm, path, missing=[99, "."])
    assert responses.loc["p3", "Q2"] != responses.loc["p3", "Q2"]  # NaN
    assert list(invalid.index) == ["p2"]
    assert responses["Q1"].max() == 2  # 99 did not become the maximum score


def test_pcm_maximum_ignores_values_that_are_not_scores(tmp_path):
    frame = pd.DataFrame({"ID": ["p1", "p2", "p3"], "Q1": [0, 2.5, 1], "Q2": [1, -1, 0]})
    path = write(tmp_path, "data.csv", frame)
    with pytest.warns(UserWarning, match="not valid scores"):
        responses, _ = loadup_pcm(path)
    assert list(responses.index) == ["p1", "p3"]  # p2 had no valid scores
    assert responses.loc["p3", "Q1"] == 1


def test_values_set_to_missing_are_named_in_one_warning(tmp_path):
    frame = pd.DataFrame(
        {"ID": ["p1", "p2", "p3", "p4"], "Q1": ["1", "x", "0.5", "1"],
         "Q2": ["2", "x", "0", "1"]}
    )
    path = write(tmp_path, "data.csv", frame)
    with pytest.warns(UserWarning) as record:
        responses, invalid = loadup_slm(path)
    messages = [str(w.message) for w in record if "not valid scores" in str(w.message)]
    assert len(messages) == 1
    message = messages[0]
    assert message.startswith("4 value(s)")
    assert "'x' (2 cell(s); items Q1, Q2)" in message
    assert "0.5 (1 cell(s); item Q1)" in message
    assert "2 (1 cell(s); item Q2)" in message
    assert list(invalid.index) == ["p2"]


@pytest.mark.parametrize("loader, kwargs", [(loadup_slm, {}), (loadup_rsm, {"max_score": 1})])
def test_long_format(tmp_path, loader, kwargs):
    long = pd.DataFrame(
        {"Person": ["p1", "p1", "p2", "p2", "p3", "p3"],
         "Item": ["Q1", "Q2"] * 3, "Score": [1, 0, 0, 1, 1, 1]}
    )
    path = write(tmp_path, "long.csv", long)
    responses, _ = load_quietly(loader, path, long=True, **kwargs)
    pd.testing.assert_frame_equal(responses, expected_wide())


def test_long_format_warns_about_repeated_responses(tmp_path):
    long = pd.DataFrame(
        {"Person": ["p1", "p1", "p1", "p2", "p2"], "Item": ["Q1", "Q1", "Q2", "Q1", "Q2"],
         "Score": [1, 0, 0, 0, 1]}
    )
    path = write(tmp_path, "long.json", long, orient="records")
    with pytest.warns(UserWarning, match=r"1 response\(s\) appear more than once.*p1, Q1"):
        responses, _ = loadup_slm(path, long=True)
    assert responses.loc["p1", "Q1"] == 0


def test_long_format_missing_codes_only_apply_to_scores(tmp_path):
    long = pd.DataFrame(
        {"Person": ["99", "99", "p2", "p2"], "Item": ["Q1", "Q2", "Q1", "Q2"],
         "Score": [1, 99, 0, 1]}
    )
    path = write(tmp_path, "long.csv", long)
    responses, _ = load_quietly(loadup_slm, path, long=True, missing=99)
    assert np.isnan(responses.loc["99", "Q2"]) and responses.loc["99", "Q1"] == 1


def mfrm_wide():
    return pd.DataFrame(
        {"Judge": ["r1", "r1", "r2", "r2"], "Person": ["p1", "p2", "p1", "p2"],
         "Q1": [2, 1, 0, 3], "Q2": [1, 1, 2, 2]}
    )


@pytest.mark.parametrize("name, kwargs", [("mfrm.csv", {}), ("mfrm.json", {"orient": "records"})])
def test_mfrm_single_facet_name(tmp_path, name, kwargs):
    path = write(tmp_path, name, mfrm_wide(), **kwargs)
    responses, _ = load_quietly(loadup_mfrm_single, path, facet_name="Judge")
    assert responses.index.names == ["Judge", "Person"]
    assert responses.loc[("r2", "p2"), "Q1"] == 3


def test_mfrm_single_default_facet_name_and_long(tmp_path):
    long = mfrm_wide().rename(columns={"Judge": "Rater"}).melt(
        id_vars=["Rater", "Person"], var_name="Item", value_name="Score")
    path = write(tmp_path, "long.csv", long)
    responses, _ = load_quietly(loadup_mfrm_single, path, long=True)
    assert responses.index.names == ["Rater", "Person"]
    assert responses.loc[("r1", "p1"), "Q2"] == 1


def test_mfrm_tabs_report_non_integers_instead_of_rounding(tmp_path):
    path = tmp_path / "tabs.xlsx"
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame({"Person": ["p1", "p2"], "Q1": [2, 1.5], "Q2": [1, "M"]}).to_excel(
            writer, sheet_name="r1", index=False)
        pd.DataFrame({"Person": ["p1", "p2"], "Q1": [0, 3], "Q2": [2, 2]}).to_excel(
            writer, sheet_name="r2", index=False)
    with pytest.warns(UserWarning, match=r"1\.5 \(1 cell\(s\); item Q1\)"):
        responses, invalid = loadup_mfrm_xlsx_tabs(str(path), max_score=3, facet_name="Judge")
    assert responses.index.names == ["Judge", "Person"]
    assert list(invalid.index) == [("r1", "p2")]  # 1.5 and 'M': no valid scores left
    responses, _ = load_quietly(
        loadup_mfrm_xlsx_tabs, str(path), max_score=3, missing=["M", 1.5])
    assert responses.index.names == ["Rater", "Person"]


def test_mfrm_multiple(tmp_path):
    files = {}
    for rater, scores in {"r1": [2, 99], "r2": [0, 3]}.items():
        frame = pd.DataFrame({"Person": ["p1", "p2"], "Q1": scores, "Q2": [1, 2]})
        files[rater] = write(tmp_path, f"{rater}.csv", frame)
    responses, _ = load_quietly(loadup_mfrm_multiple, files, max_score=3, missing=99)
    assert responses.index.names == ["Rater", "Person"]
    assert np.isnan(responses.loc[("r1", "p2"), "Q1"])


@pytest.mark.parametrize("name, kwargs", [("info.csv", {}), ("info.json", {"orient": "records"})])
def test_exogenous(tmp_path, name, kwargs):
    info = pd.DataFrame({"ID": ["p1", "p2"], "Gender": ["F", "M"], "Age": [34, 51]})
    path = write(tmp_path, name, info, **kwargs)
    exogenous = loadup_exogenous(path)
    assert exogenous.index.name == "Person"
    assert exogenous.loc["p2", "Gender"] == "M" and exogenous.loc["p1", "Age"] == 34
