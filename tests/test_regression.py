"""
pytest wrapper around run_regression_tests.py.

Same rationale as test_smoke.py. Always runs in assertion mode (never
``--generate``) — regenerating fixtures from a test run would defeat the
point of a regression suite. To regenerate fixtures deliberately, run
``python tests/run_regression_tests.py --generate`` directly and commit
the result, per that script's own docstring/CLI help.
"""

import pytest

import run_regression_tests as regression
from conftest import format_failures


CASES = [
    (model, est)
    for model in regression.RUNNERS
    for est in (regression.ESTIMATIONS if model in regression.ESTIMATED_RUNNERS else ("pair-cpat",))
]


@pytest.mark.parametrize("model, estimation", CASES, ids=[f"{m}-{e}" for m, e in CASES])
def test_regression(model, estimation):
    regression.set_estimation(estimation)
    start = len(regression.results)
    regression.RUNNERS[model](generate=False, verbose=False)
    new = regression.results[start:]

    assert new, f"{model!r} regression runner recorded no checks at all"

    failures = [(name, msg) for name, ok, msg in new if not ok]
    if failures:
        pytest.fail(format_failures(model, failures, len(new)))
