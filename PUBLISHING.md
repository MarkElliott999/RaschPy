# Publishing RaschPy to PyPI

## Prerequisites

```bash
pip install build twine
```

Create a PyPI account at https://pypi.org and generate an API token
(Account settings → API tokens → Add API token).

## Repository structure required

Before building, confirm the package layout matches what setuptools expects:

```
RaschPy/
├── raschpy/
│   ├── __init__.py          ← public API surface
│   ├── base.py
│   ├── slm.py
│   ├── pcm.py
│   ├── rsm.py
│   ├── mfrm.py
│   ├── loaders.py
│   └── simulation/
│       ├── __init__.py      ← see note below
│       ├── base_sim.py
│       ├── slm_sim.py
│       ├── pcm_sim.py
│       ├── rsm_sim.py
│       └── mfrm_sim.py
├── pyproject.toml
├── MANIFEST.in
├── README.md
├── LICENSE
└── CITATION.cff
```

> **Note:** `raschpy/simulation/` needs its own `__init__.py` so Python
> treats it as a sub-package. A minimal one is sufficient:
>
> ```python
> from raschpy.simulation.slm_sim import SLM_Sim
> from raschpy.simulation.pcm_sim import PCM_Sim
> from raschpy.simulation.rsm_sim import RSM_Sim
> from raschpy.simulation.mfrm_sim import (
>     MFRM_Sim, MFRM_Sim_Global, MFRM_Sim_Items,
>     MFRM_Sim_Thresholds, MFRM_Sim_Matrix,
> )
> ```

## Build

From the repository root:

```bash
python -m build
```

This creates two files in `dist/`:
- `raschpy-0.1.0.tar.gz` — source distribution
- `raschpy-0.1.0-py3-none-any.whl` — wheel

## Test locally before uploading

```bash
pip install dist/raschpy-0.1.0-py3-none-any.whl
python -c "from raschpy import SLM, PCM, RSM, MFRM; print('OK')"
```

## Upload to TestPyPI first (recommended)

```bash
twine upload --repository testpypi dist/*
pip install --index-url https://test.pypi.org/simple/ raschpy
```

## Upload to PyPI

```bash
twine upload dist/*
```

You will be prompted for your API token (use `__token__` as the username
and paste the token as the password), or configure `~/.pypirc`:

```ini
[pypi]
username = __token__
password = pypi-YOUR-TOKEN-HERE
```

## Subsequent releases

1. Bump the version in `pyproject.toml` and `raschpy/__init__.py`
2. Rebuild: `python -m build`
3. Upload: `twine upload dist/*`
