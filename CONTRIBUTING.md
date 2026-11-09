# Contributing to RaschPy

This covers the day-to-day development workflow: getting set up, running the
tests, code style, and how a change gets from a local edit to a release. For
*why* the codebase is shaped the way it is, see `ARCHITECTURE.md` first.

## Setup

```bash
git clone https://github.com/MarkElliott999/RaschPy.git
cd RaschPy
pip install -e ".[dev]"
```

The editable install (`-e`) is not optional, and this is worth taking
seriously: `import raschpy` resolves via whatever's first on `sys.path`, and
without an editable install it can silently resolve to a stale `pip`-installed
copy elsewhere instead of this checkout — code you're editing, tests you're
running, and results you're trusting can all quietly diverge with no error at
all. This has genuinely happened more than once on this project, including
regressing back after being fixed once already. Confirm it worked:

```bash
python -c "import raschpy; print(raschpy.__file__)"
```

That should print a path inside this checkout (`.../RaschPy/raschpy/__init__.py`),
not something under `site-packages` sitting outside it. If your workflow uses
a separate Jupyter kernel, make sure the editable install ran against the
*same* interpreter that kernel uses (`which -a python3` / the kernel's own
`sys.executable`) — installing editable into one Python and running notebooks
in another reintroduces the exact same problem.

## Running the tests

```bash
pytest                              # everything, from the repo root
pytest -k mfrm                      # just the MFRM cases
pytest --cov=raschpy --cov-report=term-missing   # with coverage
```

`pytest` from the repo root collects everything with no extra flags or
config — `tests/conftest.py` handles the `sys.path` setup. There are four
test files:

- `test_smoke.py` / `test_regression.py` / `test_edge_case.py` — pytest
  wrappers around `tests/run_smoke_tests.py` / `run_regression_tests.py` /
  `run_edge_case_tests.py`, parametrized per model. The three `run_*.py`
  scripts are also runnable directly (`python tests/run_smoke_tests.py -v`)
  if you want their own verbose CLI output, but confirm
  `raschpy.__file__` first if you do — that direct-script invocation is
  the one path `conftest.py`'s `sys.path` fallback doesn't cover.
- `test_known_issue_regressions.py` — regression tests for specific bugs
  that have already shipped broken at least once. If you fix a bug that
  was previously silent or undetected, add a test here (not just to smoke
  or regression), so the *specific* failure mode is pinned down — "some
  check somewhere still passes" isn't the same guarantee.

The regression suite asserts numeric output against fixtures committed in
`tests/regression_data/`. If a change intentionally alters output (a real
bug fix, a new default), regenerate them and commit the new fixtures:

```bash
python tests/run_regression_tests.py --generate
git status tests/regression_data/    # confirm only the expected files changed
```

Don't regenerate fixtures to make a failing test pass without first being
sure the *new* numbers are the correct ones, not the old test catching a
real regression.

CI (`.github/workflows/tests.yml`) runs the full suite against Python
3.10–3.13 on every push and PR, with a separate `flake8`/`black` lint job.
That lint job is currently `continue-on-error: true` — there's real
pre-existing formatting drift in the codebase, tracked as its own cleanup
item — so a lint failure won't block CI yet, but please don't add to the
drift in new code.

## Code style

- Library code (`raschpy/*.py`) is PEP 8, `black`-formatted (line length 88,
  matching `.flake8`'s `max-line-length`), checked with `flake8`.
- Example notebooks (`Examples/**/*.ipynb`) use a different convention on
  purpose: hanging-indent-to-bracket for multi-line calls, not
  PEP 8's bracket-on-its-own-line style. Match whichever file you're
  editing rather than imposing one style on the other.
- Docstrings are numpydoc-style (`Parameters`/`Returns`/`Attributes set`
  sections). Match the existing pattern in the file you're editing.

## Making a change

- Calibration/fit-statistics code: see `ARCHITECTURE.md`'s "Estimation
  approach" and "Known structural invariants" sections before touching
  `calibrate()`, `_build_pairwise_matrix()`, `std_errors()`, or MFRM's
  `MFRM.__init__` row-order handling — these have specific, non-obvious
  contracts that have been silently broken by well-intentioned refactors
  before.
- New public methods should follow the existing lazy-auto-calibrate
  convention (auto-trigger `calibrate()`/`fit_statistics()`/etc. if not
  already run, rather than requiring the caller to sequence calls
  manually) — see `ARCHITECTURE.md`.
- Run the full suite before proposing a change is done, using the
  editable-install-confirmed setup above — a change "verified" against a
  stale install verifies nothing.
- Commits, tags, and releases are made by the project maintainer, not
  automated as part of a code change — a finished change should be staged
  and left for review, not committed/pushed/tagged on its own.

## Release sequence

Roughly, in order (not automated end-to-end; each step needs the previous
one confirmed first):

1. **Version bump.** Update the version in three places, kept in sync:
   `pyproject.toml`, `raschpy/__init__.py`'s `__version__`, and `CHANGELOG.md`
   (new dated section, following the existing `## [X.Y.Z] - YYYY-MM-DD`
   format — `Added`/`Fixed`/`Changed`/`Documentation`/`Removed` subsections
   as applicable).
2. **Manual.** The manual's LaTeX source is maintained separately, outside
   this repository (only the built PDFs are committed here) — this step is
   maintainer-only. Save the new build as `RaschPy_manual_X_Y_Z.pdf` in
   `raschpy/docs/` (which keeps every release's manual) and in the
   repository root (which holds only the latest; remove the previous
   one there).
   Rebuild it and check it against the actual shipped code for the release
   before committing the new PDF — claims about behaviour have gone stale
   and shipped that way before, including inside already-tagged releases.
3. **Build and smoke-test.** `python -m build`, then `twine check dist/*`,
   then install the built wheel into a throwaway venv and confirm
   `from raschpy import SLM, PCM, RSM, MFRM` works — see `PUBLISHING.md`
   for the full build/upload mechanics.
4. **PyPI upload.** The actual `twine upload` (and anything else touching
   API tokens or credentials) is run by the maintainer directly, not
   automated — hand over the exact command rather than running it.
5. **GitHub.** Push the version-bump commit(s) and an annotated tag
   (`vX.Y.Z`) to `main`.
6. **conda-forge.** `recipe.yaml`'s `context.version` and `source.sha256`
   track the *last actually-published* PyPI release, not the
   in-development version — don't bump them until step 4 is genuinely
   live. In practice this is largely automated: `regro-cf-autotick-bot`
   opens a version-bump PR against `conda-forge/raschpy-feedstock` once it
   notices the new PyPI release; review and merge it rather than editing
   the recipe by hand.

Steps 1–3 are safe to prepare as part of a change; steps 4–6 are
maintainer-only and should not be run as a side effect of finishing
unrelated work.
