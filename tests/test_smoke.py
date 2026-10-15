"""
pytest wrapper around run_smoke_tests.py.

Makes the smoke suite discoverable by ``pytest`` (from the repo root, no
extra flags) instead of only via ``python tests/run_smoke_tests.py``, and
reports each model as its own test rather than one combined pass/fail
count. See conftest.py for why this matters: run_smoke_tests.py resolves
``import raschpy`` however sys.path says to, and without an editable
install that can silently be a stale pip install rather than the local
source under test.

The runner module's own ``results`` list is module-level and additive
(never cleared between calls), so each parametrized case here records
where that list stood before its own RUNNERS[model] call and only
inspects the slice appended by that call.
"""

import warnings

import pytest

import run_smoke_tests as smoke
from conftest import format_failures


@pytest.mark.parametrize("model", list(smoke.RUNNERS))
def test_smoke(model):
    start = len(smoke.results)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        smoke.RUNNERS[model](verbose=False)
    new = smoke.results[start:]

    assert new, f"{model!r} smoke runner recorded no checks at all"

    failures = [(name, msg) for name, ok, msg in new if not ok]
    if failures:
        pytest.fail(format_failures(model, failures, len(new)))
