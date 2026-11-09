"""RSM and PCM CML with unused score categories (v2.0.0): the width ridge with only two categories used, and the
warning when the CML 'null' estimates of the steps between the used categories do not exist."""
import warnings

import numpy as np
import pandas as pd
import pytest

import raschpy as rp


def _draws(cls, K, unused, N, seed, I=10):
    """RSM (PCM: item threshold deviations SD 0.3) responses drawn with the unused categories' probabilities set to
    zero; returns (responses, true item locations)."""
    rng = np.random.default_rng(seed)
    d = rng.uniform(-1.5, 1.5, I)
    d -= d.mean()
    tau = np.sort(rng.uniform(-2.5, 2.5, K))
    tau -= tau.mean()
    dev = rng.normal(0, 0.3, (I, K)) if cls == "PCM" else np.zeros((I, K))
    dev -= dev.mean(1, keepdims=True)
    B = d[:, None] + tau[None, :] + dev
    th = rng.normal(0, 1.5, N)
    cum = np.concatenate([np.zeros((I, 1)), np.cumsum(B, 1)], 1)
    lw = np.arange(K + 1)[None, None, :] * th[:, None, None] - cum[None, :, :]
    lw[..., unused] = -np.inf
    p = np.exp(lw - lw.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    X = pd.DataFrame((rng.random((N, I, 1)) > np.cumsum(p, -1)).sum(-1).astype(float),
                     columns=[f"I{i}" for i in range(I)])
    return X, B.mean(1) - B.mean()


@pytest.mark.parametrize("cls", ["RSM", "PCM"])
def test_ridge_with_only_two_categories_used_converges(cls):
    """Maximum score 3 with categories 1 and 2 unused: every width is next to an unused category and the likelihood
    pushes them all down; the ridge pulls each (PCM: each group's mean widths) towards 0, so the fit converges with
    accurate item locations."""
    X, loc = _draws(cls, 3, [1, 2], 150, 90000 + 50000 * (cls == "PCM") + 3000 + 500)
    m = rp.RSM(X, max_score=3) if cls == "RSM" else rp.PCM(X, max_score_vector=[3] * 10)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.calibrate(estimation="cml")
    assert m.convergence["converged"]
    est = m.items.to_numpy(float) - m.items.mean()
    assert np.sqrt(np.mean((est - loc) ** 2)) < 0.2


def test_pcm_null_steps_that_do_not_exist_are_reported():
    """Categories 1 and 2 unused (maximum score 4): for these data the conditional likelihood is flat along the
    items' steps where the fit stops, so the CML 'null' estimates do not exist; the warning says so, not 'did not
    converge', and 'ridge' converges."""
    X, _ = _draws("PCM", 4, [1, 2], 150, 90000 + 50000 + 4000 + 500 + 93)
    m = rp.PCM(X, max_score_vector=[4] * 10)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m.calibrate(estimation="cml", unused_categories="null")
    msgs = " ".join(str(x.message) for x in w)
    assert "CML estimates of the steps" in msgs and "did not converge" not in msgs
    r = rp.PCM(X, max_score_vector=[4] * 10)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r.calibrate(estimation="cml")
    assert r.convergence["converged"]
