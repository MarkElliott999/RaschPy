"""
pytest wrapper around run_edge_case_tests.py.

Same rationale as test_smoke.py. Edge-case checks report a third status,
``'xfail'`` (a known, currently-expected failure the runner deliberately
records rather than hiding — see that script's own ``check``/``xfail``
helpers), alongside ``'pass'``/``'fail'``. Only ``'fail'`` should fail
this test; a present ``'xfail'`` doesn't fail or skip it (the check did
run, and xfail is its expected outcome) but is surfaced as a pytest
warning so a newly-fixed case doesn't go unnoticed indefinitely.
"""

import warnings

import pytest

import run_edge_case_tests as edge
from conftest import format_failures


@pytest.mark.parametrize("model", list(edge.RUNNERS))
def test_edge_case(model):
    start = len(edge.results)
    edge.RUNNERS[model](verbose=False)
    new = edge.results[start:]

    assert new, f"{model!r} edge-case runner recorded no checks at all"

    failures = [(name, msg) for name, status, msg in new if status == "fail"]
    if failures:
        pytest.fail(format_failures(model, failures, len(new)))

    xfailed = [name for name, status, _ in new if status == "xfail"]
    if xfailed:
        warnings.warn(
            f"{len(xfailed)}/{len(new)} known xfail checks for {model!r} "
            f"(see run_edge_case_tests.py): {', '.join(xfailed)}",
            stacklevel=1,
        )
