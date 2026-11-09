"""Progress reporting (v2.0.0): progress= on the slow public methods, the in-place progress line on stderr, nested
loops kept silent, and the long-run warning."""
import inspect
import warnings

import pytest

import raschpy as rp
from raschpy import _progress
from raschpy.simulation import MFRM_Sim, PCM_Sim


def _pcm():
    sim = PCM_Sim(no_of_items=6, no_of_persons=200, max_score_vector=[3] * 6, seed=1)
    m = rp.PCM(sim.responses)
    m.calibrate(estimation="cml")
    return m


def test_progress_argument_and_docstring():
    for meth in (rp.SLM.calibrate, rp.PCM.std_errors, rp.RSM.model_selection, rp.MFRM.std_errors,
                 rp.MFRM.per_rater_model_selection, rp.PCM.global_fit):
        p = inspect.signature(meth).parameters["progress"]
        assert p.default is None and p.kind == p.KEYWORD_ONLY
        doc = meth.__doc__
        assert "progress : {None, True, False}" in doc
        if "Returns\n" in doc and "Parameters\n" in doc:
            assert doc.index("progress :") < doc.index("Returns\n")
    with pytest.raises(ValueError, match="progress must be"):
        _pcm().std_errors(se_method="bootstrap", no_of_samples=2, progress="yes")


def test_progress_line_shown_suppressed_and_not_nested(capsys):
    m = _pcm()
    m.std_errors(se_method="bootstrap", no_of_samples=10, seed=1, progress=True)
    err = capsys.readouterr().err
    assert "Bootstrap standard errors: 10/10 resamples" in err and "done" in err
    assert "CML fit" not in err  # the resamples' own fits stay silent
    m.std_errors(se_method="bootstrap", no_of_samples=10, seed=1, progress=False)
    assert capsys.readouterr().err == ""
    m.std_errors(se_method="bootstrap", no_of_samples=10, seed=1)  # fast: None shows nothing
    assert capsys.readouterr().err == ""
    rp.PCM(m.responses).calibrate(estimation="cml", progress=True)
    assert "CML fit:" in capsys.readouterr().err


def test_long_run_warning(monkeypatch, capsys):
    monkeypatch.setattr(_progress, "LONG", 0.0)
    m = _pcm()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        m.std_errors(se_method="bootstrap", no_of_samples=5, seed=1, progress=False)
    msgs = [str(w.message) for w in rec if "estimated run time" in str(w.message)]
    assert len(msgs) == 1 and "se_method='sandwich'" in msgs[0]
    assert capsys.readouterr().err == ""


def test_mfrm_model_selection_progress(capsys):
    sim = MFRM_Sim(no_of_items=5, no_of_persons=100, max_score=3, no_of_facet_elements=3, model="items", seed=2)
    m = rp.MFRM(sim.responses, max_score=3, validate=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.model_selection(progress=True)
    assert "Model selection: candidate fits: 8/8 models" in capsys.readouterr().err
