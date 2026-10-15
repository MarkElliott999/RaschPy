"""
Shared pytest configuration for RaschPy's test suite.

Ensures the repo root is importable, so ``import raschpy`` in
run_smoke_tests.py / run_regression_tests.py / run_edge_case_tests.py
resolves to the locally checked-out source rather than a stale pip
install shadowing it elsewhere on sys.path — see the "Test environment"
note in CONTRIBUTING.md. The recommended, CI-enforced setup is still
``pip install -e .`` from the repo root; this path insertion is a
belt-and-braces fallback for running pytest without that.

Also puts tests/ itself on sys.path so the test_*.py wrappers below can
``import run_smoke_tests`` etc. as plain modules, without needing an
``__init__.py`` / package layout under tests/.
"""

import os
import sys

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_TESTS_DIR)

for _path in (_REPO_ROOT, _TESTS_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)


def format_failures(model, failures, total):
    """
    Render a list of (name, msg) check failures as one pytest.fail() message.

    Shared by test_smoke.py / test_regression.py / test_edge_case.py, whose
    underlying runners all report checks as (name, ok_or_status, msg) tuples
    appended to a module-level ``results`` list.
    """
    lines = []
    for name, msg in failures:
        short = msg.strip().splitlines()[-1] if msg else ""
        lines.append(f"  - {name}" + (f": {short}" if short else ""))
    return f"{len(failures)}/{total} checks failed for {model!r}:\n" + "\n".join(lines)
